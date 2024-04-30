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
use tracing::debug;

#[derive(Debug)]
pub struct StagingBuffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    device: Weak<super::Device>,
}

impl StagingBuffer {
    #[tracing::instrument(skip(r), name = "new_staging_buffer")]
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
        debug!(requirements = ?reqs, "got buffer memory requirements");
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

    #[tracing::instrument(
        skip(self, data),
        fields(
            data_len = %data.len(),
            buffer_len = %self.size
        )
    )]
    pub fn upload(&self, data: &[u8], offset: vk::DeviceSize) -> Result<(), Error<'static>> {
        debug!("upload");
        let device = self.device.upgrade()
            .expect("device destroyed before staging buffer");
        let mut mem = device.map_memory_(self.memory, offset, self.size, vk::MemoryMapFlags::empty())
            .vk("vk_map_memory")?;
        let copy_size = std::cmp::min(mem.len(), data.len());
        mem[..copy_size].copy_from_slice(&data[..copy_size]);
        Ok(())
    }

    #[inline(always)]
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.size as _
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
    staging_buffer: &StagingBuffer,
    image: &super::InnerImage,
    size: [u32; 2],
    offset: vk::DeviceSize,
    stride: u32,
) -> Result<(), Error<'static>> {
    let buffer = staging_buffer.handle();
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::empty());
    vk_call!(device, begin_command_buffer(cb, &begin_info))?;
    let queue_idx = queue.index as u32;
    let buf_barrier = vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .src_queue_family_index(queue_idx)
        .dst_queue_family_index(queue_idx)
        .buffer(buffer)
        .offset(offset)
        .size(staging_buffer.size)
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
        let copies = [
            vk::BufferImageCopy {
                buffer_offset: offset,
                buffer_row_length: size[0],
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
        ];
        debug!(?copies, "copying buffer to image");
        device.cmd_copy_buffer_to_image(
            cb,
            buffer, image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copies,
        );
    }
    vk_call!(device, end_command_buffer(cb))
}
