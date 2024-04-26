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
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout,
        final_layout,
    }
}

#[derive(Debug)]
pub struct RenderSetup {
    device: Arc<super::Device>,
    pub format: vk::Format,
    handle: vk::RenderPass,
    pub usage_flags: vk::ImageUsageFlags,
}

impl RenderSetup {
    pub fn new(
        device: Arc<super::Device>,
        color_format: vk::Format,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> Result<Self, Error<'static>> {
        // let create_info = vk::PipelineLayoutCreateInfo::builder()
        //     .flags(vk::PipelineLayoutCreateFlags::empty())
        //     .set_layouts(&[])
        //     .push_constant_ranges(&[]);
        // let layout = unsafe {
        //     device.create_pipeline_layout(&create_info, None)
        // }
        //     .vk("vkCreatePipelineLayout")
        //     .map(|handle| scopeguard::guard(handle, |handle| unsafe {
        //         device.destroy_pipeline_layout(handle, None);
        //     }))?;

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
        }
    }
}

#[inline(always)]
fn usage_flags_always() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT
    | vk::ImageUsageFlags::TRANSFER_DST
}
