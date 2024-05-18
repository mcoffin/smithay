use cgmath::{
    Matrix4,
    Vector2,
};
use ash::vk;
use crate::backend::allocator::vulkan::format::{
    FormatMapping,
    FORMAT_MAPPINGS,
};
use super::{
    shaders::{
        SOURCES as SHADERS,
        ShaderModule,
        ColorTransform,
    },
    Error,
    ErrorExt,
};
use std::{
    mem::{
        size_of,
        offset_of,
    },
    num::NonZeroI32,
    ptr,
    sync::Arc,
};
use scopeguard::ScopeGuard;
use tracing::debug;

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
    pub format: FormatMapping,
    handle: vk::RenderPass,
    pub usage_flags: vk::ImageUsageFlags,
    layouts: [PipelineLayout; 2],
    pipelines: [vk::Pipeline; 2],
}

static PUSH_CONSTANT_RANGES: [vk::PushConstantRange; 2] = [
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

const fn spec_map_entry<T: Sized>(
    constant_id: usize,
    offset: usize
) -> vk::SpecializationMapEntry {
    vk::SpecializationMapEntry {
        constant_id: constant_id as _,
        offset: offset as _,
        size: size_of::<T>(),
    }
}

trait FormatMappingExt {
    fn color_transform(&self) -> ColorTransform;
}

impl FormatMappingExt for FormatMapping {
    #[inline]
    fn color_transform(&self) -> ColorTransform {
        if self.has_srgb() {
            ColorTransform::GammaToLinear
        } else {
            ColorTransform::Identity
        }
    }
}

impl RenderSetup {
    pub fn new(
        device: Arc<super::Device>,
        color_format: FormatMapping,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> Result<Self, Error<'static>> {
        let info = vk::PipelineLayoutCreateInfo::builder()
            .flags(vk::PipelineLayoutCreateFlags::empty())
            .set_layouts(&[])
            .push_constant_ranges(&PUSH_CONSTANT_RANGES)
            .build();
        let quad_layout = unsafe {
            device.create_pipeline_layout(&info, None)
        }
            .vk("vkCreatePipelineLayout")
            .map(|v| scopeguard::guard(v, |v| unsafe {
                device.destroy_pipeline_layout(v, None);
            }))?;
        let tex_layout = PipelineLayout::new(&device, vk::Filter::LINEAR, vk::Filter::LINEAR)
            .map(|v| scopeguard::guard(v, |v| unsafe {
                v.destroy(&device);
            }))?;

        let create_shader_module = |source: &[u8]| {
            ShaderModule::new(&device, source).vk("vkCreateShaderModule")
        };
        let vert_module = create_shader_module(SHADERS.vert.common)?;
        let frag_module_quad = create_shader_module(SHADERS.frag.quad)?;
        let quad_shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module.handle())
                .name(c"main")
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module_quad.handle())
                .name(c"main")
                .build(),
        ];
        let frag_module_tex = create_shader_module(SHADERS.frag.texture)?;
        // TODO: derive from format
        let specialization = spec_map_entry::<ColorTransform>(0, 0);
        let specialization = vk::SpecializationInfo {
            map_entry_count: 1,
            p_map_entries: (&specialization as *const vk::SpecializationMapEntry).cast(),
            data_size: size_of::<ColorTransform>(),
            p_data: (&color_format.color_transform() as *const ColorTransform).cast(),
        };
        let tex_shader_stages = [
            quad_shader_stages[0],
            vk::PipelineShaderStageCreateInfo {
                module: frag_module_tex.handle(),
                p_specialization_info: (&specialization as *const vk::SpecializationInfo).cast(),
                ..quad_shader_stages[1]
            },
        ];

        const DYNAMIC_STATES: &[vk::DynamicState] = &[
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
        ];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(DYNAMIC_STATES);
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .build();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_FAN)
            .primitive_restart_enable(false)
            .build();
        const VIEWPORT_STATE: vk::PipelineViewportStateCreateInfo = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            viewport_count: 1,
            p_viewports: ptr::null(),
            scissor_count: 1,
            p_scissors: ptr::null(),
        };
        const RASTERIZATION_STATE: vk::PipelineRasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0f32,
            depth_bias_clamp: 0f32,
            depth_bias_slope_factor: 0f32,
            line_width: 1f32,
        };
        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1f32)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();
        static COLOR_BLEND_ATTACHMENTS: &[vk::PipelineColorBlendAttachmentState] = &[
            color_blend_attachment(true, true),
        ];
        let blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(COLOR_BLEND_ATTACHMENTS)
            .blend_constants([0f32; 4]);

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
        let fmt = color_format.srgb_or_default();
        let color_attachments = [
            color_attachment(
                fmt,
                initial_layout,
                final_layout,
            )
        ];
        debug!(?color_attachments, "creating render pass");
        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&color_attachments)
            .subpasses(&subpasses);
        let device = device.clone();
        let render_pass = unsafe {
            device.create_render_pass(&create_info, None)
                .vk("vkCreateRenderPass")
                .map(|v| scopeguard::guard(v, |v| {
                    device.destroy_render_pass(v, None);
                }))
        }?;

        let quad_info = vk::GraphicsPipelineCreateInfo::builder()
            .flags(vk::PipelineCreateFlags::empty())
            .stages(&quad_shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&VIEWPORT_STATE)
            .rasterization_state(&RASTERIZATION_STATE)
            .multisample_state(&multisampling)
            .color_blend_state(&blend_state)
            .dynamic_state(&dynamic_state)
            .layout(*quad_layout)
            .render_pass(*render_pass)
            .subpass(0)
            .build();
        let infos = [
            quad_info,
            vk::GraphicsPipelineCreateInfo {
                stage_count: tex_shader_stages.len() as _,
                p_stages: tex_shader_stages.as_ptr(),
                layout: tex_layout.layout,
                ..quad_info
            },
        ];
        let pipelines = unsafe {
            let mut pipelines = [vk::Pipeline::null(); 2];
            let result = (device.fp_v1_0().create_graphics_pipelines)(
                device.handle(),
                vk::PipelineCache::null(),
                infos.len() as _,
                infos.as_ptr(),
                ptr::null(),
                pipelines.as_mut_ptr(),
            );
            NonZeroI32::new(result.as_raw()).map_or_else(
                || Ok(scopeguard::guard(pipelines, |a| {
                    let it = a.into_iter().filter(|&v| v != vk::Pipeline::null());
                    for v in it {
                        device.destroy_pipeline(v, None);
                    }
                })),
                |e| Err(Error::Vk {
                    context: "vkCreateGraphicsPipelines",
                    result: vk::Result::from_raw(e.get()),
                })
            )
        }?;

        Ok(RenderSetup {
            format: color_format,
            handle: ScopeGuard::into_inner(render_pass),
            usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            layouts: [
                From::from(ScopeGuard::into_inner(quad_layout)),
                ScopeGuard::into_inner(tex_layout),
            ],
            pipelines: ScopeGuard::into_inner(pipelines),
            device,
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

    #[inline(always)]
    fn pipeline_at(&self, index: usize) -> (vk::Pipeline, &PipelineLayout) {
        (self.pipelines[index], &self.layouts[index])
    }

    #[inline(always)]
    pub(super) fn quad_pipeline(&self) -> (vk::Pipeline, &PipelineLayout) {
        self.pipeline_at(0)
    }

    #[inline(always)]
    pub(super) fn tex_pipeline(&self) -> (vk::Pipeline, &PipelineLayout) {
        self.pipeline_at(1)
    }
}

impl Drop for RenderSetup {
    fn drop(&mut self) {
        unsafe {
            for &pipeline in &self.pipelines {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device.destroy_render_pass(self.handle, None);
            for &layout in &self.layouts {
                layout.destroy(&self.device);
            }
        }
    }
}

#[inline(always)]
fn usage_flags_always() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT
    | vk::ImageUsageFlags::TRANSFER_DST
}

const fn color_blend_attachment(blend_enabled: bool, premultiplied: bool) -> vk::PipelineColorBlendAttachmentState {
    type AttState = vk::PipelineColorBlendAttachmentState;
    const PREMULTIPLIED: AttState = AttState {
        blend_enable: vk::TRUE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    };
    const COLOR_BLEND_ATTACHMENT_OPAQUE: AttState = AttState {
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    };
    match (blend_enabled, premultiplied) {
        (false, ..) => COLOR_BLEND_ATTACHMENT_OPAQUE,
        (true, true) => PREMULTIPLIED,
        (true, false) => AttState {
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            ..PREMULTIPLIED
        },
    }
}

const fn color_attachment(
    format: vk::Format,
    initial_layout: vk::ImageLayout,
    final_layout: vk::ImageLayout,
) -> vk::AttachmentDescription {
    vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format: format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::LOAD,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout,
        final_layout,
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PipelineLayout {
    pub layout: vk::PipelineLayout,
    pub ds_layout: vk::DescriptorSetLayout,
    pub sampler: vk::Sampler,
}

impl PipelineLayout {
    fn new(
        device: &ash::Device,
        upscale_filter: vk::Filter,
        downscale_filter: vk::Filter,
    ) -> Result<Self, Error<'static>> {
        let info = vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: upscale_filter,
            min_filter: downscale_filter,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            mip_lod_bias: 0f32,
            anisotropy_enable: vk::FALSE,
            max_anisotropy: 0f32,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::NEVER,
            min_lod: 0f32,
            max_lod: 0.25,
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            unnormalized_coordinates: vk::FALSE,
        };
        let sampler = unsafe {
            device.create_sampler(&info, None)
                .vk("vkCreateSampler")
                .map(|v| scopeguard::guard(v, |v| {
                    device.destroy_sampler(v, None);
                }))
        }?;
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: &*sampler as *const _,
            },
        ];
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings);
        let ds_layout = unsafe {
            device.create_descriptor_set_layout(&info, None)
                .vk("vkCreateDescriptorSetLayout")
                .map(|v| scopeguard::guard(v, |v| {
                    device.destroy_descriptor_set_layout(v, None);
                }))
        }?;

        let ds_layouts = [*ds_layout];
        let info = vk::PipelineLayoutCreateInfo::builder()
            .flags(vk::PipelineLayoutCreateFlags::empty())
            .set_layouts(&ds_layouts)
            .push_constant_ranges(&PUSH_CONSTANT_RANGES)
            .build();
        let layout = unsafe {
            device.create_pipeline_layout(&info, None)
                .vk("vkCreatePipelineLayout")
        }?;
        Ok(PipelineLayout {
            layout,
            ds_layout: ScopeGuard::into_inner(ds_layout),
            sampler: ScopeGuard::into_inner(sampler),
        })
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_pipeline_layout(self.layout, None);
        if self.ds_layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.ds_layout, None);
        }
        if self.sampler != vk::Sampler::null() {
            device.destroy_sampler(self.sampler, None);
        }
    }
}

impl From<vk::PipelineLayout> for PipelineLayout {
    #[inline(always)]
    fn from(handle: vk::PipelineLayout) -> Self {
        PipelineLayout {
            layout: handle,
            ds_layout: vk::DescriptorSetLayout::null(),
            sampler: vk::Sampler::null(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn uniform_data_offset_matches() {
        assert_eq!(offset_of!(UniformData, frag), 80);
    }
}
