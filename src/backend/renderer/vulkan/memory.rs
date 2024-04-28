use ash::vk;
use core::ops::{
    Deref,
    DerefMut,
};

pub trait DeviceExtMemory {
    fn map_memory_(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        flags: vk::MemoryMapFlags,
    ) -> Result<MappedMemory<'_>, vk::Result>;
}

impl DeviceExtMemory for ash::Device {
    fn map_memory_(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        flags: vk::MemoryMapFlags,
    ) -> Result<MappedMemory<'_>, vk::Result> {
        unsafe {
            let ptr = self.map_memory(memory, offset, size, flags)?;
            Ok(MappedMemory {
                data: std::slice::from_raw_parts_mut(ptr.cast(), size as _),
                handle: memory,
                device: self,
            })
        }
    }
}

pub struct MappedMemory<'a> {
    data: &'a mut [u8],
    handle: vk::DeviceMemory,
    device: &'a ash::Device,
}

impl<'a> Deref for MappedMemory<'a> {
    type Target = [u8];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &*self.data
    }
}

impl<'a> DerefMut for MappedMemory<'a> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl<'a> Drop for MappedMemory<'a> {
    fn drop(&mut self) {
        self.data = &mut [];
        unsafe {
            self.device.unmap_memory(self.handle);
        }
    }
}
