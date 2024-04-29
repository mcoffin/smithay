use ash::vk;
use scopeguard::ScopeGuard;
use super::{
    render_pass,
    Error,
    ErrorExt,
    InnerImage,
};

#[derive(Debug)]
pub struct InnerImageView {
    image_view: vk::ImageView,
    ds_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl InnerImageView {
    pub fn new(image: &InnerImage, layout: &render_pass::PipelineLayout, max: u32) -> Result<Self, Error<'static>> {
        let device = image.device.upgrade().unwrap();
        let info = vk::ImageViewCreateInfo::builder()
            .image(image.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image.format.vk)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: if image.has_alpha() {
                    vk::ComponentSwizzle::IDENTITY
                } else {
                    vk::ComponentSwizzle::ONE
                },
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let mut ret = unsafe {
            device.create_image_view(&info, None)
                .vk("vkCreateImageView")
        }.map(|image_view| InnerImageView {
            image_view,
            ds_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
        }).map(|v| scopeguard::guard(v, |mut v| unsafe {
            v.destroy(&device);
        }))?;

        let pool_sizes = &[
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: max,
            },
        ];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max)
            .pool_sizes(pool_sizes);
        ret.ds_pool = unsafe {
            device.create_descriptor_pool(&info, None)
        }.vk("vkCreateDescriptorPool")?;

        let ds_layouts = vec![layout.ds_layout; max as usize];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(ret.ds_pool)
            .set_layouts(ds_layouts.as_slice());
        ret.descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&info)
        }.vk("vkAllocateDescriptorSets")?;
        debug_assert_eq!(ret.descriptor_sets.len(), max as usize);

        let image_info = [
            vk::DescriptorImageInfo {
                sampler: layout.sampler,
                image_view: ret.image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        ];
        let mut writes = Vec::with_capacity(ret.descriptor_sets.len());
        let it = ret.descriptor_sets.iter()
            .map(|&ds| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(ds)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info)
                    .build()
            });
        writes.extend(it);
        unsafe {
            device.update_descriptor_sets(writes.as_slice(), &[]);
        }

        Ok(ScopeGuard::into_inner(ret))
    }
    pub(super) unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.image_view != vk::ImageView::null() {
            device.destroy_image_view(self.image_view, None);
        }
        if self.ds_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.ds_pool, None);
        }
    }

    #[inline(always)]
    pub fn descriptor_sets(&self) -> &[vk::DescriptorSet] {
        self.descriptor_sets.as_slice()
    }
}
