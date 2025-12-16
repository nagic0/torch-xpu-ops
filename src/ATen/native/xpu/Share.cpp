#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/native/Share.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>

namespace at::native::xpu {

static c10::Storage storage_share_xpu(const c10::Storage& src, const c10::Device& device) {
  // Get source storage information
  void* src_ptr = src.mutable_data();
  size_t nbytes = src.nbytes();

  auto sycl_device = ::at::xpu::getCurrentXPUStream(device.index()).queue().get_device();
  auto sycl_context = ::at::xpu::getCurrentXPUStream(device.index()).queue().get_context();

  // Ensure the context is a unified runtime context
  TORCH_CHECK(
      sycl_context.get_platform().get_backend() == sycl::backend::ext_oneapi_level_zero,
      "The backend is not Level Zero. USM Import requires Level Zero backend.");

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

  auto* deleter_context = new DeleterContext{src_impl, src_ptr, device, sycl_context};

  c10::DeleterFnPtr deleter = [](void* ctx) {
    auto* context = static_cast<DeleterContext*>(ctx);
    // First, update the PyTorch metadata in src to reflect that sharing ended
    context->src_impl->mutable_data_ptr().remove_usm_device(context->device);
    // Then, release the imported USM memory
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

  auto data_ptr = c10::DataPtr(src_ptr, deleter_context, deleter, device);
  data_ptr.enable_usm_support();

  auto new_storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      std::move(data_ptr),
      c10::GetAllocator(c10::DeviceType::XPU),
      /* resizable */ false);

  // Update src metadata to record that it's being shared with a new device
  src.unsafeGetStorageImpl()->mutable_data_ptr().add_usm_device(device);

  return c10::Storage(std::move(new_storage_impl));
}

} // namespace at::native::xpu

namespace at::native {
REGISTER_XPU_DISPATCH(share_stub, &xpu::storage_share_xpu);
} // namespace at::native
