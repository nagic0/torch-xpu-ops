#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/native/UsmShare.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>

namespace at::native::xpu {

static c10::Storage storage_usm_share_xpu(const c10::Storage& src, const c10::Device& device) {
  // Set device using DeviceGuard
  c10::DeviceGuard device_guard(device);
  
  // Normalize device to actual device index (handle device.index() == -1 case)
  c10::Device actual_device = device_guard.current_device();

  // Get source storage information
  void* src_ptr = src.mutable_data();
  size_t nbytes = src.nbytes();

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
      "usm_share_xpu: target device does not support USM (not integrated GPU): ",
      actual_device);

  try {
      sycl::ext::oneapi::experimental::prepare_for_device_copy(
          src_ptr, nbytes, sycl_context);
  } catch (const sycl::exception& e) {
      TORCH_CHECK(false, "SYCL USM Import failed: ", e.what());
  }

  // Create a new storage with a custom deleter that also updates src metadata
  c10::StorageImpl* src_impl = src.unsafeGetStorageImpl();
  // Increment ref count of src to prevent it from being freed while the new
  // storage shares its data. The ref count will be decremented in the deleter.
  c10::raw::intrusive_ptr::incref(src_impl);

  struct DeleterContext {
    c10::StorageImpl* src_impl{};
    void* data{};
    c10::Device device;
    sycl::context sycl_ctx;
  };

  auto* deleter_context = new DeleterContext{src_impl, src_ptr, actual_device, sycl_context};

  c10::DeleterFnPtr deleter = [](void* ctx) {
    auto* context = static_cast<DeleterContext*>(ctx);
    // Release the imported USM memory
    try {
        sycl::ext::oneapi::experimental::release_from_device_copy(
            context->data, context->sycl_ctx);
    } catch (const std::exception& e) {
        TORCH_WARN("Error releasing USM import: ", e.what());
    }
    // Decrement the ref count of src
    c10::raw::intrusive_ptr::decref(context->src_impl);
    delete context;
  };

  auto data_ptr = c10::DataPtr(src_ptr, deleter_context, deleter, actual_device);

  auto new_storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      std::move(data_ptr),
      c10::GetAllocator(c10::DeviceType::XPU),
      /* resizable */ false);

  return c10::Storage(std::move(new_storage_impl));
}

} // namespace at::native::xpu

namespace at::native {
REGISTER_XPU_DISPATCH(usm_share_stub, &xpu::storage_usm_share_xpu);
} // namespace at::native
