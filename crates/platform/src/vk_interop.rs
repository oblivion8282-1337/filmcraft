//! Imports GPU memory a camera-RAW bridge exported (a CUDA allocation shared as a POSIX file
//! descriptor) into the app's wgpu Vulkan device, so its frames reach the compositor without a
//! host round trip ([ADR 0002](../../../docs/adr/0002-gpu-frame-interop.md)). The rest of FilmCraft
//! sees a safe [`filmcraft_gpu::external::ExternalImporter`]; every Vulkan call is here.
//!
//! An import is `vkCreateBuffer` (external memory, opaque fd) + `vkAllocateMemory` importing the fd
//! into a device-local memory type + `vkBindBufferMemory`, wrapped into a `wgpu::Buffer` that frees
//! both when wgpu is done with it. Anything unexpected — another backend, another GPU (UUID), the
//! extension not enabled, a driver refusal — returns `None`, and the compositor uploads the frame
//! from its CPU fallback instead.

use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::sync::Arc;

use ash::vk;
use filmcraft_frame::ExternalMemory;
use wgpu::hal::api::Vulkan;

/// The importer for one process (stateless: everything it needs comes with the device).
struct VkImporter;

/// `deviceUUID` of the physical device behind a wgpu device, when it runs on Vulkan.
fn device_uuid(device: &wgpu::Device) -> Option<[u8; 16]> {
    // SAFETY: `as_hal` hands out the device's hal object for the guard's lifetime; we only read
    // its handles (instance, physical device) and make one query that has no side effects.
    unsafe {
        let hal = device.as_hal::<Vulkan>()?;
        let instance = hal.shared_instance().raw_instance();
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        instance.get_physical_device_properties2(hal.raw_physical_device(), &mut props);
        Some(id.device_uuid)
    }
}

/// Whether wgpu enabled `VK_KHR_external_memory_fd` on the device (it does when the driver has it).
fn has_external_memory_fd(device: &wgpu::Device) -> bool {
    // SAFETY: reading the list of extensions the hal device was created with.
    unsafe { device.as_hal::<Vulkan>().is_some_and(|hal| hal.enabled_device_extensions().contains(&ash::khr::external_memory_fd::NAME)) }
}

impl filmcraft_gpu::external::ExternalImporter for VkImporter {
    fn import(&self, device: &wgpu::Device, memory: &ExternalMemory) -> Option<wgpu::Buffer> {
        if device_uuid(device)? != memory.device_uuid() || !has_external_memory_fd(device) || memory.size() == 0 {
            return None;
        }
        let fd = memory.handle().downcast_ref::<OwnedFd>()?.try_clone().ok()?;
        let (buffer, mem) = import_buffer(device, fd, memory.size())?;
        let hal_device = {
            // SAFETY: as above; the cloned `ash::Device` is a handle table, valid while the device
            // lives, and the drop callback below runs before wgpu destroys the device.
            unsafe { device.as_hal::<Vulkan>()?.raw_device().clone() }
        };
        // SAFETY: `buffer` was created on this device and bound to `mem`, which backs its whole
        // size; the callback destroys both exactly once, when wgpu drops the hal buffer (after
        // the GPU is done with it). wgpu never maps an externally owned buffer.
        let hal_buffer = unsafe {
            wgpu::hal::vulkan::Buffer::from_raw_externally_owned(
                buffer,
                Box::new(move || {
                    hal_device.destroy_buffer(buffer, None);
                    hal_device.free_memory(mem, None);
                }),
            )
        };
        let desc = wgpu::BufferDescriptor {
            label: Some("filmcraft-external-frame"),
            size: memory.size(),
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };
        // SAFETY: the hal buffer comes from this device, matches `desc` (size, a copy source,
        // not mapped) and its contents are initialised by the exporter.
        Some(unsafe { device.create_buffer_from_hal::<Vulkan>(hal_buffer, &desc) })
    }
}

/// Import `fd` (`size` bytes, the exact exported size) as a buffer bound to its memory.
fn import_buffer(device: &wgpu::Device, fd: OwnedFd, size: u64) -> Option<(vk::Buffer, vk::DeviceMemory)> {
    // SAFETY: every handle used is the device's own, alive for the guard's lifetime; each object
    // created here is destroyed again on every failure path, and `fd` is owned: Vulkan takes it
    // over only when the import succeeds, otherwise it is closed here.
    unsafe {
        let hal = device.as_hal::<Vulkan>()?;
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let buffer = raw.create_buffer(&info, None).ok()?;
        let req = raw.get_buffer_memory_requirements(buffer);
        let props = instance.get_physical_device_memory_properties(hal.raw_physical_device());
        let kind = (0..props.memory_type_count.min(vk::MAX_MEMORY_TYPES as u32)).find(|&i| {
            req.memory_type_bits & (1 << i) != 0
                && props.memory_types.get(i as usize).is_some_and(|t| t.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
        });
        let Some(kind) = kind.filter(|_| req.size <= size) else {
            raw.destroy_buffer(buffer, None);
            return None;
        };
        let raw_fd = fd.into_raw_fd();
        let mut import = vk::ImportMemoryFdInfoKHR::default().handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD).fd(raw_fd);
        let alloc = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(kind).push_next(&mut import);
        let mem = match raw.allocate_memory(&alloc, None) {
            Ok(m) => m,
            Err(e) => {
                // the import failed: the descriptor is still ours to close
                drop(OwnedFd::from_raw_fd(raw_fd));
                raw.destroy_buffer(buffer, None);
                log::warn!("importing a GPU frame buffer ({size} bytes) failed: {e}");
                return None;
            }
        };
        if let Err(e) = raw.bind_buffer_memory(buffer, mem, 0) {
            raw.destroy_buffer(buffer, None);
            raw.free_memory(mem, None);
            log::warn!("binding an imported GPU frame buffer failed: {e}");
            return None;
        }
        Some((buffer, mem))
    }
}

/// Register the Vulkan importer for `device` (the app's wgpu device) and announce its GPU, so
/// camera-RAW sources ask their bridges for GPU frames. Whether the device can import (it runs on
/// Vulkan with `VK_KHR_external_memory_fd`).
pub fn register(device: &wgpu::Device) -> bool {
    let Some(uuid) = device_uuid(device).filter(|_| has_external_memory_fd(device)) else {
        log::info!("GPU frame import unavailable (not Vulkan, or no VK_KHR_external_memory_fd): camera-RAW frames take the host path");
        return false;
    };
    filmcraft_gpu::external::register_importer(Arc::new(VkImporter));
    filmcraft_frame::external::set_import_device(Some(uuid));
    log::info!("GPU frame import on device {}", uuid.iter().map(|b| format!("{b:02x}")).collect::<String>());
    true
}
