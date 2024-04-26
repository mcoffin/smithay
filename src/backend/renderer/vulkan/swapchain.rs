use ash::vk;
use super::{
    VulkanRenderer,
    Error,
    ErrorExt,
};
use scopeguard::ScopeGuard;
use std::{
    cell::Cell,
    fmt,
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            Ordering,
        },
    },
};
use tracing::{warn, trace};

pub struct Swapchain {
    device: Arc<super::Device>,
    handle: vk::SwapchainKHR,
    ext: ash::extensions::khr::Swapchain,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    extent: vk::Extent2D,
    images: Box<[SwapchainImage]>,
    pub(crate) submit_image: Cell<SubmitImage>,
}

impl Swapchain {
    pub fn with_surface(
        r: &VulkanRenderer,
        surface_info: SurfaceInfo<'_>,
        fallback_extent: &vk::Extent2D,
        render_setup: &super::RenderSetup,
    ) -> Result<Self, Error<'static>> {
        use ash::extensions as vk_ext;
        let SurfaceInfo { surface, surface_ext, color_space, capabilities, .. } = surface_info;
        let device = r.device.clone();
        let swapchain_ext = vk_ext::khr::Swapchain::new(r.instance(), &device);
        let format = render_setup.format;
        let extent = capabilities.extent_or(fallback_extent);
        let present_family_idx = r.queues.present_queue(r.physical_device(), surface, surface_ext)?;
        let mut queue_family_indices: &[u32] = &[
            present_family_idx as _,
            r.queues.graphics.index as _,
        ];
        assert_eq!(present_family_idx, r.queues.graphics.index);
        if present_family_idx == r.queues.graphics.index {
            queue_family_indices = &queue_family_indices[..1];
        }
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .flags(vk::SwapchainCreateFlagsKHR::empty())
            .surface(surface)
            .min_image_count(capabilities.image_count())
            .image_format(format)
            .image_color_space(color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(render_setup.swapchain_usage_flags())
            .image_sharing_mode(if queue_family_indices.len() > 1 { vk::SharingMode::CONCURRENT } else { vk::SharingMode::EXCLUSIVE })
            .queue_family_indices(queue_family_indices)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(surface_info.present_mode);
        trace!("create_swapchain: {:#?}", &*create_info);
        let handle = unsafe {
            swapchain_ext.create_swapchain(&create_info, None)
        }
            .map_err(SwapchainError::vk("vkCreateSwapchainKHR"))
            .map(|v| scopeguard::guard(v, |swapchain| unsafe {
                swapchain_ext.destroy_swapchain(swapchain, None);
            }))?;
        let images = unsafe {
            swapchain_ext.get_swapchain_images(*handle)
        }.vk("vkGetSwapchainImagesKHR")?;
        let mut swap_images = scopeguard::guard(Vec::<SwapchainImage>::with_capacity(images.len()), |images| unsafe {
            for swap_img in images.iter() {
                swap_img.destroy(&device);
            }
        });
        for img in images.into_iter() {
            let swap_image = SwapchainImage::new(r.device(), img, format, render_setup.render_pass(), &extent)?;
            swap_images.push(swap_image);
        }
        Ok(Swapchain {
            handle: ScopeGuard::into_inner(handle),
            ext: swapchain_ext,
            format: vk::SurfaceFormatKHR {
                format: render_setup.format,
                color_space,
            },
            present_mode: surface_info.present_mode,
            extent,
            images: ScopeGuard::into_inner(swap_images).into(),
            device,
            submit_image: Cell::new(SubmitImage::default()),
        })
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn extension(&self) -> &ash::extensions::khr::Swapchain {
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

    pub(super) fn acquire(
        &self,
        timeout: u64,
        feedback: AcquireFeedback,
    ) -> Result<(u32, &SwapchainImage), AcquireError<'_>> {
        let AcquireFeedback { semaphore, fence } = feedback;
        let (idx, suboptimal) = unsafe {
            self.ext.acquire_next_image(
                self.handle,
                timeout,
                semaphore,
                fence,
            )
        }.map_err(SwapchainError::vk("vkAcquireNextImageKHR"))?;
        let ret = (idx, &self.images()[idx as usize]);
        if suboptimal {
            Err(AcquireError::Suboptimal(ret))
        } else {
            Ok(ret)
        }
    }

    pub(super) fn submit(&self, r: &VulkanRenderer) -> Result<bool, SwapchainError<&'static str>> {
        let img = self.submit_image.get();
        let swapchains = [self.handle()];
        let images = [img.index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(img.wait_semaphores())
            .swapchains(&swapchains)
            .image_indices(&images);
        unsafe {
            self.ext.queue_present(r.queues.graphics.handle, &present_info)
        }.map_err(SwapchainError::vk("vkQueuePresentKHR"))
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
            for swap_image in self.images() {
                swap_image.destroy(&self.device);
            }
            self.ext.destroy_swapchain(self.handle, None);
        }
    }
}

#[derive(Debug)]
pub struct SupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SupportDetails {
    pub fn with_surface(
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
    pub fn valid(&self) -> bool {
        !self.formats.is_empty() && !self.present_modes.is_empty()
    }
    pub fn choose_format(&self) -> Option<vk::SurfaceFormatKHR> {
        const SUPPORTED_FORMATS: &[vk::Format] = &[
            vk::Format::B8G8R8A8_SRGB,
        ];
        SUPPORTED_FORMATS.iter().find_map(|&fmt| {
            self.formats.iter()
                .find(|&&vk::SurfaceFormatKHR { format, color_space }| format == fmt && color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .copied()
        })
    }
    pub fn choose_present_mode(&self) -> vk::PresentModeKHR {
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
                height: fallback.height.clamp(
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

#[derive(Debug)]
pub struct SwapchainImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
    pub image_ready_semaphore: vk::Semaphore,
    pub submit_semaphore: vk::Semaphore,
    pub transitioned: AtomicBool,
}

impl SwapchainImage {
    fn new(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        render_pass: vk::RenderPass,
        extent: &vk::Extent2D,
    ) -> Result<Self, SwapchainError<&'static str>> {
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
        }
            .map_err(SwapchainError::vk("vkCreateImageView"))
            .map(|v| scopeguard::guard(v, |v| unsafe {
                device.destroy_image_view(v, None);
            }))?;
        let fb_attachments = [
            *image_view,
        ];
        let fb_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&fb_attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        let framebuffer = unsafe {
            device.create_framebuffer(&fb_info, None)
        }
            .map_err(SwapchainError::vk("vkCreateFramebuffer"))
            .map(|v| scopeguard::guard(v, |v| unsafe {
                device.destroy_framebuffer(v, None);
            }))?;
        let create_semaphore = || {
            let sem_info = vk::SemaphoreCreateInfo::builder()
                .flags(vk::SemaphoreCreateFlags::empty())
                .build();
            unsafe {
                device.create_semaphore(&sem_info, None)
            }
                .map_err(SwapchainError::vk("vkCreateSemaphore"))
                .map(|v| scopeguard::guard(v, |v| unsafe {
                    device.destroy_semaphore(v, None);
                }))
        };
        let image_ready_sem = create_semaphore()?;
        let submit_sem = create_semaphore()?;
        Ok(SwapchainImage {
            image,
            image_view: ScopeGuard::into_inner(image_view),
            framebuffer: ScopeGuard::into_inner(framebuffer),
            image_ready_semaphore: ScopeGuard::into_inner(image_ready_sem),
            submit_semaphore: ScopeGuard::into_inner(submit_sem),
            transitioned: AtomicBool::new(false),
        })
    }
    #[inline]
    unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_framebuffer(self.framebuffer, None);
        device.destroy_image_view(self.image_view, None);
        let semaphores = [
            self.image_ready_semaphore,
            self.submit_semaphore,
        ];
        for sem in semaphores {
            device.destroy_semaphore(sem, None);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SwapchainError<S> {
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

/// Helper struct for [`Swapchain::from_surface`] to reduce parameter-spam
#[derive(Clone, Copy)]
pub struct SurfaceInfo<'a> {
    pub surface: vk::SurfaceKHR,
    pub surface_ext: &'a ash::extensions::khr::Surface,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub capabilities: &'a vk::SurfaceCapabilitiesKHR,
}

impl<'a> SurfaceInfo<'a> {
    /// Convenience function for building [`SurfaceInfo`] from a reference to [`SupportDetails`] if
    /// you already have one
    #[inline(always)]
    pub fn new(
        surface: vk::SurfaceKHR,
        surface_ext: &'a ash::extensions::khr::Surface,
        color_space: vk::ColorSpaceKHR,
        support: &'a SupportDetails
    ) -> Self {
        SurfaceInfo {
            surface,
            surface_ext,
            color_space,
            present_mode: support.choose_present_mode(),
            capabilities: &support.capabilities,
        }
    }
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum AcquireError<'a> {
    #[error(transparent)]
    Vk(#[from] SwapchainError<&'static str>),
    #[error("suboptimal swapchain")]
    Suboptimal((u32, &'a SwapchainImage)),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcquireFeedback {
    pub semaphore: vk::Semaphore,
    pub fence: vk::Fence,
}

impl Default for AcquireFeedback {
    #[inline]
    fn default() -> Self {
        AcquireFeedback {
            semaphore: vk::Semaphore::null(),
            fence: vk::Fence::null(),
        }
    }
}

impl From<vk::Semaphore> for AcquireFeedback {
    #[inline(always)]
    fn from(semaphore: vk::Semaphore) -> Self {
        AcquireFeedback {
            semaphore,
            ..Default::default()
        }
    }
}

impl From<vk::Fence> for AcquireFeedback {
    #[inline(always)]
    fn from(fence: vk::Fence) -> Self {
        AcquireFeedback {
            fence,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmitImage {
    pub index: u32,
    pub wait_semaphores: [vk::Semaphore; 1],
}

impl SubmitImage {
    pub const fn new(index: u32) -> Self {
        SubmitImage {
            index,
            wait_semaphores: [vk::Semaphore::null()],
        }
    }
    #[inline(always)]
    fn wait_semaphores(&self) -> &[vk::Semaphore] {
        if self.wait_semaphores[0] != vk::Semaphore::null() {
            &self.wait_semaphores[..]
        } else {
            &[]
        }
    }
}

impl Default for SubmitImage {
    #[inline(always)]
    fn default() -> Self {
        SubmitImage {
            index: 0,
            wait_semaphores: [vk::Semaphore::null()],
        }
    }
}
