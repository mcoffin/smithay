use ash::vk;
use super::{
    VulkanRenderer,
    Error,
    ErrorExt,
};
use scopeguard::ScopeGuard;
use std::{
    fmt,
    sync::Arc,
};

pub struct Swapchain {
    device: Arc<super::Device>,
    handle: vk::SwapchainKHR,
    ext: ash::extensions::khr::Swapchain,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    extent: vk::Extent2D,
    images: Box<[SwapchainImage]>,
}

impl Swapchain {
    pub fn with_surface(
        r: &VulkanRenderer,
        surface: vk::SurfaceKHR,
        surface_ext: &ash::extensions::khr::Surface,
        fallback_extent: &vk::Extent2D,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, Error<'static>> {
        let device = r.device.clone();
        let swapchain_ext = ash::extensions::khr::Swapchain::new(r.instance(), &device);
        let support = SupportDetails::with_surface(r.phd.handle(), surface, surface_ext)
            .map_err(Error::from)
            .and_then(|v| if v.valid() {
                Ok(v)
            } else {
                Err(Error::SwapchainSupport)
            })?;
        let vk::SurfaceFormatKHR { format, color_space } = support.choose_format()
            .ok_or(Error::SwapchainFormat)?;
        let present_mode = support.choose_present_mode();
        let extent = support.capabilities.extent_or(fallback_extent);
        let present_family_idx = r.queues.present_queue(r.phd.handle(), surface, surface_ext)?;
        let mut queue_family_indices: &[u32] = &[
            present_family_idx as _,
            r.queues.graphics.index as _,
        ];
        assert_eq!(present_family_idx, r.queues.graphics.index);
        if present_family_idx == r.queues.graphics.index {
            queue_family_indices = &queue_family_indices[..1];
        }
        let info = vk::SwapchainCreateInfoKHR::builder()
            .flags(vk::SwapchainCreateFlagsKHR::empty())
            .surface(surface)
            .min_image_count(support.capabilities.image_count())
            .image_format(format)
            .image_color_space(color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(if queue_family_indices.len() > 1 { vk::SharingMode::CONCURRENT } else { vk::SharingMode::EXCLUSIVE })
            .queue_family_indices(queue_family_indices)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode);
        let handle = unsafe {
            swapchain_ext.create_swapchain(&info, None)
        }
            .map_err(SwapchainError::vk("vkCreateSwapchainKHR"))
            .map(|v| scopeguard::guard(v, |swapchain| unsafe {
                swapchain_ext.destroy_swapchain(swapchain, None);
            }))?;
        let images = unsafe {
            swapchain_ext.get_swapchain_images(*handle)
        }.vk("vkGetSwapchainImagesKHR")?;
        let mut swap_images = scopeguard::guard(Vec::with_capacity(images.len()), |images| unsafe {
            for &SwapchainImage { image_view, .. } in images.iter() {
                device.destroy_image_view(image_view, None);
            }
        });
        for img in images.into_iter() {
            let swap_image = SwapchainImage::new(r.device(), img, format)?;
            swap_images.push(swap_image);
        }
        Ok(Swapchain {
            handle: ScopeGuard::into_inner(handle),
            ext: swapchain_ext,
            format: vk::SurfaceFormatKHR { format, color_space },
            present_mode,
            extent,
            images: ScopeGuard::into_inner(swap_images).into(),
            device,
        })
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn extension(&self) -> &ash::extensions::khr::Swapchain {
        &self.ext
    }

    #[inline(always)]
    pub fn extent(&self) -> &vk::Extent2D {
        &self.extent
    }

    #[inline(always)]
    pub fn images(&self) -> &[SwapchainImage] {
        &self.images
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn surface_format(&self) -> &vk::SurfaceFormatKHR {
        &self.format
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn format(&self) -> vk::Format {
        self.format.format
    }
}

impl fmt::Debug for Swapchain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Swapchain")
            .field("handle", &self.handle)
            .field("format", &self.format)
            .field("present_mode", &self.present_mode)
            .field("extent", &self.extent)
            .field("images", &&*self.images)
            .finish()
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            for &SwapchainImage { image_view, .. } in self.images() {
                self.device.destroy_image_view(image_view, None);
            }
            self.ext.destroy_swapchain(self.handle, None);
        }
    }
}

#[derive(Debug)]
struct SupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SupportDetails {
    fn with_surface(
        pd: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        ext: &ash::extensions::khr::Surface
    ) -> Result<Self, SwapchainError<&'static str>> {
        let caps = unsafe {
            ext.get_physical_device_surface_capabilities(pd, surface)
        }.map_err(SwapchainError::vk("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))?;
        let formats = unsafe {
            ext.get_physical_device_surface_formats(pd, surface)
        }.map_err(SwapchainError::vk("vkGetPhysicalDeviceSurfaceFormatsKHR"))?;
        let present_modes = unsafe {
            ext.get_physical_device_surface_present_modes(pd, surface)
        }.map_err(SwapchainError::vk("vkGetPhysicalDeviceSurfacePresentModesKHR"))?;
        Ok(SupportDetails {
            capabilities: caps,
            formats,
            present_modes,
        })
    }
    fn valid(&self) -> bool {
        !self.formats.is_empty() && !self.present_modes.is_empty()
    }
    fn choose_format(&self) -> Option<vk::SurfaceFormatKHR> {
        const SUPPORTED_FORMATS: &[vk::Format] = &[
            vk::Format::B8G8R8A8_SRGB,
        ];
        SUPPORTED_FORMATS.iter().find_map(|&fmt| {
            self.formats.iter()
                .find(|&&vk::SurfaceFormatKHR { format, color_space }| format == fmt && color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .copied()
        })
    }
    fn choose_present_mode(&self) -> vk::PresentModeKHR {
        const MODES: &[vk::PresentModeKHR] = &[
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::FIFO_RELAXED,
        ];
        let supported = self.present_modes.as_slice();
        MODES.iter()
            .find(|&mode| supported.contains(mode))
            .copied()
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }
}

trait CapabilitiesExt {
    fn extent(&self) -> Option<&vk::Extent2D>;
    fn extent_or(&self, fallback: &vk::Extent2D) -> vk::Extent2D;
    fn image_count(&self) -> u32;
}

impl CapabilitiesExt for vk::SurfaceCapabilitiesKHR {
    #[inline(always)]
    fn extent(&self) -> Option<&vk::Extent2D> {
        Some(&self.current_extent)
            .filter(|v| v.width != u32::MAX)
    }
    fn extent_or(&self, fallback: &vk::Extent2D) -> vk::Extent2D {
        self.extent()
            .copied()
            .unwrap_or_else(|| vk::Extent2D {
                width: fallback.width.clamp(
                    self.min_image_extent.width,
                    self.max_image_extent.width,
                ),
                height: fallback.width.clamp(
                    self.min_image_extent.height,
                    self.max_image_extent.height,
                ),
            })
    }
    fn image_count(&self) -> u32 {
        use std::num::NonZeroU32;
        let ret = self.min_image_count + 1;
        NonZeroU32::new(self.max_image_count)
            .map_or(ret, |max_count| std::cmp::min(ret, max_count.get()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapchainImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
}

impl SwapchainImage {
    fn new(device: &ash::Device, image: vk::Image, format: vk::Format) -> Result<Self, SwapchainError<&'static str>> {
        const IDENTITY_MAPPING: vk::ComponentMapping = vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        };
        let info = vk::ImageViewCreateInfo::builder()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(IDENTITY_MAPPING)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let image_view = unsafe {
            device.create_image_view(&info, None)
        }.map_err(SwapchainError::vk("vkCreateImageView"))?;
        Ok(SwapchainImage {
            image,
            image_view,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct SwapchainError<S> {
    context: S,
    error: vk::Result,
}

impl<S> SwapchainError<S> {
    #[inline(always)]
    fn vk(context: S) -> impl FnOnce(vk::Result) -> Self {
        move |error| SwapchainError {
            context,
            error,
        }
    }
}

impl<S: fmt::Display> fmt::Display for SwapchainError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:?}", &self.context, self.error)
    }
}

impl<S> std::error::Error for SwapchainError<S>
where
    S: fmt::Display + fmt::Debug,
{}

impl<'a> From<SwapchainError<&'a str>> for Error<'a> {
    #[inline(always)]
    fn from(e: SwapchainError<&'a str>) -> Self {
        Error::Vk {
            context: e.context,
            result: e.error,
        }
    }
}
