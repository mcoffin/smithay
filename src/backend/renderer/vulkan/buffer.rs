use ash::vk;
use crate::vk_create_guarded;
use std::{
    num::NonZeroI32,
    sync::Arc,
};
use super::{
    memory::DeviceExtMemory,
    Error,
    ErrorExt,
};
use scopeguard::ScopeGuard;

macro_rules! vk_call {
    ($dev:expr, $fn_name:ident ($($arg:expr),+ $(,)?)) => {
        unsafe {
            $dev.$fn_name($($arg),+)
                .vk(concat!("vk_", stringify!($create_fn)))
        }
    };
}

#[derive(Debug)]
pub struct StagingBuffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    device: Arc<super::Device>,
}

impl StagingBuffer {
    pub fn new(
        r: &super::VulkanRenderer,
        size: vk::DeviceSize,
        sharing_mode: vk::SharingMode,
    ) -> Result<Self, Error<'static>> {
        const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_SRC;
        let device = r.device();
        let info = vk::BufferCreateInfo::builder()
            .flags(vk::BufferCreateFlags::empty())
            .size(size)
            .usage(USAGE)
            .sharing_mode(sharing_mode)
            .build();
        let buffer = vk_create_guarded!(device, create_buffer(&info, None), destroy_buffer(None))?;
        let reqs = unsafe { device.get_buffer_memory_requirements(*buffer) };
        let host_props = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mem_idx = r.find_memory_type_r(reqs.memory_type_bits, host_props)?;
        let allocate_memory = |size: vk::DeviceSize, idx: u32| {
            let info = vk::MemoryAllocateInfo::builder()
                .allocation_size(size)
                .memory_type_index(idx)
                .build();
            vk_create_guarded!(device, allocate_memory(&info, None), free_memory(None))
        };
        let mem = allocate_memory(reqs.size, mem_idx)?;
        vk_call!(device, bind_buffer_memory(*buffer, *mem, 0))?;

        Ok(StagingBuffer {
            handle: ScopeGuard::into_inner(buffer),
            memory: ScopeGuard::into_inner(mem),
            size,
            device: r.device.clone(),
        })
    }

    pub fn upload(&self, data: &[u8], offset: vk::DeviceSize) -> Result<(), Error<'static>> {
        let mut mem = self.device.map_memory_(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
            .vk("vk_map_memory")?;
        let copy_size = std::cmp::min(mem.len() - offset as usize, data.len());
        mem[..copy_size].copy_from_slice(&data[..copy_size]);
        Ok(())
    }

    #[inline(always)]
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    #[inline(always)]
    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.size as _
    }

    #[inline(always)]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }
}

trait RendererExt {
    fn find_memory_type_r(&self, ty: u32, props: vk::MemoryPropertyFlags) -> Result<u32, Error<'static>>;
}

impl RendererExt for super::VulkanRenderer {
    #[inline(always)]
    fn find_memory_type_r(&self, ty: u32, props: vk::MemoryPropertyFlags) -> Result<u32, Error<'static>> {
        self.find_memory_type(ty, props)
            .ok_or(Error::NoMemoryType {
                bits: ty,
                flags: props,
            })
            .map(|(idx, ..)| idx as _)
    }
}
