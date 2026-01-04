#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>

#include <level_zero/ze_api.h>

constexpr uint32_t ZEX_HOST_MEM_ALLOC_FLAG_USE_HOST_PTR = ZE_BIT(30);

// Macro to check Level Zero API call results
#define L0_SAFE_CALL(call)                                  \
    {                                                       \
        ze_result_t status = (call);                        \
        if (status != ZE_RESULT_SUCCESS) {                  \
          TORCH_CHECK(false, "Level Zero API call failed, error code: ", status); \
        }                                                   \
    }

namespace at::native {

static void ze_share(
  ze_context_handle_t context,
  ze_device_handle_t device,
  void* shared_buffer,
  size_t buffer_size,
  size_t alignment
) {
  ze_device_mem_alloc_desc_t usm_device_desc = {};
  ze_host_mem_alloc_desc_t usm_host_desc = {};
  usm_host_desc.flags = ZEX_HOST_MEM_ALLOC_FLAG_USE_HOST_PTR;
  L0_SAFE_CALL(zeMemAllocShared(context, &usm_device_desc, &usm_host_desc, buffer_size, alignment, device, &shared_buffer));
}

// Implementation for XPU backend
// self: dummy tensor on XPU device (carries device info)
// src:  CPU storage containing the data (passed by value to match codegen signature)
Tensor usm_share_from_xpu(const Tensor& self, c10::Storage src) {
  void* src_ptr = src.data_ptr().get();
  size_t nbytes = src.nbytes();
  c10::Device target_device = self.device();

  TORCH_CHECK(
    src.device().is_cpu(),
    "usm_share_from_xpu: source tensor must be on CPU, got: ",
    src.device());
  TORCH_CHECK(
      target_device.type() == c10::DeviceType::XPU,
      "usm_share_from_xpu: target device must be XPU, got: ",
      target_device);

  // Handle empty case
  if (nbytes == 0) {
    return at::empty_like(self, self.options());
  }

  // Set device using DeviceGuard
  c10::DeviceGuard device_guard(target_device);
  c10::Device actual_device = device_guard.current_device();

  // Get the actual SYCL device and context
  sycl::device& sycl_device = c10::xpu::get_raw_device(actual_device.index());
  auto sycl_context = ::at::xpu::getCurrentXPUStream(actual_device.index()).queue().get_context();

  // Ensure the context is a unified runtime context
  TORCH_CHECK(
      sycl_context.get_platform().get_backend() == sycl::backend::ext_oneapi_level_zero,
      "The backend is not Level Zero. USM Import requires Level Zero backend.");

  // Check device USM support (integrated GPU with host unified memory)
  TORCH_CHECK(
      sycl_device.get_info<sycl::info::device::host_unified_memory>(),
      "usm_share_from_xpu: target device does not support USM (not integrated GPU): ",
      actual_device);

  // get ze_context_handle_t
  ze_context_handle_t ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
  ze_device_handle_t ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

  L0_SAFE_CALL(zeContextGetStatus(ze_context));
  L0_SAFE_CALL(zeDeviceGetStatus(ze_device));

  ze_share(ze_context, ze_device, src_ptr, nbytes, 65536);

  // Create a new storage with a custom deleter that also updates src metadata
  c10::StorageImpl* src_impl = src.unsafeGetStorageImpl();
  // Increment ref count of src to prevent it from being freed while the new
  // storage shares its data. The ref count will be decremented in the deleter.
  c10::raw::intrusive_ptr::incref(src_impl);

  struct DeleterContext {
    c10::StorageImpl* src_impl{};
    ze_context_handle_t ze_context{};
    void* data{};
    c10::Device device;
  };

  auto* deleter_context =
      new DeleterContext{src_impl, ze_context, src_ptr, actual_device};

  c10::DeleterFnPtr deleter = [](void* ctx) {
    auto* context = static_cast<DeleterContext*>(ctx);
    L0_SAFE_CALL(zeMemFree(context->ze_context, context->data));
    // Decrement the ref count of src
    c10::raw::intrusive_ptr::decref(context->src_impl);
    delete context;
  };

  auto data_ptr = c10::DataPtr(src_ptr, deleter_context, deleter, actual_device);

  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      std::move(data_ptr),
      c10::GetAllocator(c10::DeviceType::XPU),
      /* resizable */ false);

  return at::empty({0}, self.options()).set_(std::move(storage_impl));
}

} // namespace at::native
