use ash::vk;
use crate::{
    vk_call,
    vk_create_guarded,
};
use std::sync::{
    Arc,
    Weak,
};
use super::{
    memory::DeviceExtMemory,
    Error,
    ErrorExt,
};
use scopeguard::ScopeGuard;

#[derive(Debug)]
pub struct StagingBuffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    device: Weak<super::Device>,
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
            device: Arc::downgrade(&r.device),
        })
    }

    pub fn upload(&self, data: &[u8], offset: vk::DeviceSize) -> Result<(), Error<'static>> {
        let device = self.device.upgrade()
            .expect("device destroyed before staging buffer");
        let mut mem = device.map_memory_(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
            .vk("vk_map_memory")?;
        let copy_size = std::cmp::min(mem.len() - offset as usize, data.len());
        mem[..copy_size].copy_from_slice(&data[..copy_size]);
        Ok(())
    }

    #[inline(always)]
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        let device = self.device.upgrade()
            .expect("device destroyed before staging buffer");
        unsafe {
            device.destroy_buffer(self.handle, None);
            device.free_memory(self.memory, None);
        }
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

pub fn record_copy_buffer_image(
    device: &ash::Device,
    queue: &super::Queue,
    cb: vk::CommandBuffer,
    buffer: vk::Buffer,
    image: &super::InnerImage,
    size: [u32; 2],
    offset: vk::DeviceSize,
) -> Result<(), Error<'static>> {
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::empty());
    vk_call!(device, begin_command_buffer(cb, &begin_info))?;
    let queue_idx = queue.index as u32;
    let p_size: vk::DeviceSize = match image.format.vk {
        vk::Format::B8G8R8_SRGB
        | vk::Format::R8G8B8_SRGB => 3,
        vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB => 4,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        _ => unimplemented!(),
    };
    let full_size = (size[0] as vk::DeviceSize) * p_size * (size[1] as vk::DeviceSize);
    let buf_barrier = vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .src_queue_family_index(queue_idx)
        .dst_queue_family_index(queue_idx)
        .buffer(buffer)
        .offset(0)
        .size(full_size)
        .build();
    let img_barriers = [
        vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(image.layout.get())
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(queue_idx)
            .dst_queue_family_index(queue_idx)
            .image(image.image)
            .subresource_range(super::COLOR_SINGLE_LAYER)
            .build(),
    ];
    unsafe {
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[], &[buf_barrier], &img_barriers,
        );
        device.cmd_copy_buffer_to_image(
            cb,
            buffer, image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[
                vk::BufferImageCopy {
                    buffer_offset: offset as _,
                    buffer_row_length: p_size as u32 * size[0],
                    buffer_image_height: size[1],
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: size[0],
                        height: size[1],
                        depth: 1,
                    },
                }
            ]
        );
    }
    vk_call!(device, end_command_buffer(cb))
}
