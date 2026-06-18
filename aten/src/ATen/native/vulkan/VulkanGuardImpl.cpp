#include <ATen/native/vulkan/api/api.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

namespace at {
namespace detail {

namespace {

namespace api = at::native::vulkan::api;

struct VulkanGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  VulkanGuardImpl() = default;

  // NOLINTNEXTLINE
  explicit VulkanGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::Vulkan);
  }

  DeviceType type() const override {
    return DeviceType::Vulkan;
  }

  // Thread-local device index. -1 means uninitialized.
  static int& current_device_index() {
    thread_local int index = -1;
    return index;
  }

  Device exchangeDevice(Device dev) const override {
    auto old = getDevice();
    current_device_index() = static_cast<int>(dev.index());
    // Ensure runtime is initialized on first device access
    (void)api::runtime();
    return old;
  }

  Device getDevice() const override {
    auto idx = current_device_index();
    if (idx < 0) {
      // First access: initialize runtime and pick default device
      (void)api::runtime();
      idx = static_cast<int>(api::runtime()->default_adapter_i());
      current_device_index() = idx;
    }
    return Device(DeviceType::Vulkan, idx);
  }

  void setDevice(Device dev) const override {
    exchangeDevice(dev);
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    current_device_index() = static_cast<int>(d.index());
  }

  Stream getStream(Device d) const noexcept override {
    (void)d;
    // Return default stream (no-op until stream management is implemented)
    return Stream(Stream::DEFAULT, getDevice());
  }

  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    (void)s;
    return getStream(getDevice());
  }

  DeviceIndex deviceCount() const noexcept override {
    // Lazy init: ensure runtime exists
    (void)api::runtime();
    return static_cast<DeviceIndex>(api::runtime()->num_adapters());
  }

  bool isEnabled() const noexcept override {
    return true;
  }

  // Event functions - not yet supported
  void record(
      void** /*event*/,
      const Stream& /*stream*/,
      const DeviceIndex /*device_index*/,
      const c10::EventFlag /*flag*/) const override {
    TORCH_CHECK(false, "Vulkan: recordEvent not yet implemented");
  }

  void block(void* /*event*/, const Stream& /*stream*/) const override {
    TORCH_CHECK(false, "Vulkan: block not yet implemented");
  }

  bool queryEvent(void* /*event*/) const override {
    TORCH_CHECK(false, "Vulkan: queryEvent not yet implemented");
  }

  void synchronizeEvent(void* /*event*/) const override {
    TORCH_CHECK(false, "Vulkan: synchronizeEvent not yet implemented");
  }

  void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
      const noexcept override {}
};

} // namespace

C10_EXPORT_DEFINE_DEVICEGUARDIMPL(VulkanGuardImpl);

} // namespace detail
} // namespace at
