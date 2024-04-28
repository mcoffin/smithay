use cgmath::{
    Matrix4,
    Vector2,
};
use ash::vk;
use super::{
    VulkanRenderer,
    Error,
    ErrorExt,
};
use std::sync::Arc;
use scopeguard::ScopeGuard;

const fn color_attachment(
    format: vk::Format,
    initial_layout: vk::ImageLayout,
    final_layout: vk::ImageLayout,
) -> vk::AttachmentDescription {
    vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::LOAD,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout,
        final_layout,
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub union UniformDataFrag {
    pub alpha: f32,
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct UniformDataVert<S> {
    pub transform: Matrix4<S>,
    pub tex_offset: Vector2<S>,
    pub tex_extent: Vector2<S>,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct UniformData {
    pub vert: UniformDataVert<f32>,
    pub frag: UniformDataFrag,
}

#[derive(Debug)]
pub struct RenderSetup {
    device: Arc<super::Device>,
    pub format: vk::Format,
    handle: vk::RenderPass,
    pub usage_flags: vk::ImageUsageFlags,
    quad_pipeline_layout: vk::PipelineLayout,
}

impl RenderSetup {
    pub fn new(
        device: Arc<super::Device>,
        color_format: vk::Format,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> Result<Self, Error<'static>> {
        use std::mem::{size_of, offset_of};
        const PUSH_CONSTANT_RANGES: [vk::PushConstantRange; 2] = [
            vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: offset_of!(UniformData, vert) as _,
                size: size_of::<UniformDataVert<f32>>() as _,
            },
            vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                offset: offset_of!(UniformData, frag) as _,
                size: size_of::<UniformDataFrag>() as _,
            },
        ];
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .flags(vk::PipelineLayoutCreateFlags::empty())
            .set_layouts(&[])
            .push_constant_ranges(&PUSH_CONSTANT_RANGES);
        let quad_layout = unsafe {
            device.create_pipeline_layout(&create_info, None)
        }
            .vk("vkCreatePipelineLayout")
            .map(|v| scopeguard::guard(v, |v| unsafe {
                device.destroy_pipeline_layout(v, None);
            }))?;

        let color_attachments = [
            vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }
        ];
        let subpasses = [
            vk::SubpassDescription::builder()
                .flags(vk::SubpassDescriptionFlags::empty())
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachments)
                .build(),
        ];
        let color_attachments = [
            color_attachment(
                color_format,
                initial_layout,
                final_layout,
            )
        ];
        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&color_attachments)
            .subpasses(&subpasses);
        let device = device.clone();
        unsafe {
            device.create_render_pass(&create_info, None)
        }
            .vk("vkCreateRenderPass")
            .map(move |v| RenderSetup {
                device,
                format: color_format,
                handle: v,
                usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                quad_pipeline_layout: ScopeGuard::into_inner(quad_layout),
            })
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn render_pass(&self) -> vk::RenderPass {
        self.handle
    }

    #[inline]
    pub fn swapchain_usage_flags(&self) -> vk::ImageUsageFlags {
        self.usage_flags | usage_flags_always()
    }
}

impl Drop for RenderSetup {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_render_pass(self.handle, None);
            self.device.destroy_pipeline_layout(self.quad_pipeline_layout, None);
        }
    }
}

#[inline(always)]
fn usage_flags_always() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT
    | vk::ImageUsageFlags::TRANSFER_DST
}
