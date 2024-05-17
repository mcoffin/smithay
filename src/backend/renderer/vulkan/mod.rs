//! Implementation of the rendering traits using Vulkan 1.1+
//!
//! Good entry points are [`VulkanRenderer`] and [`VulkanFrame`]

use self::buffer::StagingBuffer;

use super::{ImportDma, ImportDmaWl, ImportMem, ImportMemWl};
use crate::{
    backend::{
        allocator::{
            dmabuf::{Dmabuf, WeakDmabuf},
            Buffer, Format as DrmFormat, Fourcc, Modifier as DrmModifier,
            vulkan::format::{
                FormatMapping,
                FORMAT_MAPPINGS,
            },
        },
        drm::DrmNode,
        renderer::{sync::SyncPoint, DebugFlags, Frame, Renderer, Texture, TextureFilter},
        vulkan::{
            util::{OwnedHandle, VulkanHandle},
            version::Version,
            PhysicalDevice,
        },
    },
    contextual_handles, fn_name,
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
    vulkan_handles,
    wayland::compositor,
    wayland::shm::{
        with_buffer_contents,
        BufferData,
    },
};
use ash::{
    extensions::{ext::ImageDrmFormatModifier, khr::ExternalMemoryFd},
    prelude::VkResult,
    vk,
};
use cgmath::Vector2;
use scopeguard::ScopeGuard;
use wayland_server::protocol::wl_shm;
use std::{
    borrow::Cow, cell::{
        Cell,
        RefCell,
    },
    collections::{
        HashMap,
        LinkedList,
        linked_list,
    },
    fmt, mem, num::NonZeroU64, ops::{Deref, DerefMut}, os::fd::{AsRawFd, IntoRawFd}, rc::Rc, sync::{
        atomic::Ordering, mpsc, Arc, Weak
    }
};

#[allow(unused_imports)]
use tracing::{debug, error, trace, warn};

mod buffer;
mod dmabuf;
mod fence;
mod format;
mod frame;
mod memory;
mod queue;
mod render_pass;
mod shaders;
mod swapchain;
mod transform;
mod util;
mod view;

use util::*;
use fence::*;
use swapchain::Swapchain;
use render_pass::{
    RenderSetup,
    UniformData,
    UniformDataVert,
    UniformDataFrag,
};
use transform::TransformExt;
use queue::{
    QueueFamilies,
    Queue,
};
use format::{
    Format,
    FormatInfo,
};
use view::InnerImageView;
pub use frame::VulkanFrame;
use frame::SubmittedFrame;

/// [`Renderer`] implementation using the [`vulkan`](https://www.vulkan.org/) graphics API
#[derive(Debug)]
pub struct VulkanRenderer {
    phd: PhysicalDevice,
    device: Arc<Device>,
    queues: QueueFamilies,
    _node: Option<DrmNode>,
    formats: HashMap<Fourcc, FormatInfo>,
    dmabuf_formats: Rc<[DrmFormat]>,
    extensions: Extensions,
    debug_flags: DebugFlags,
    dmabuf_cache: HashMap<WeakDmabuf, VulkanImage>,
    shm_images: Vec<WeakVulkanImage>,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    command_pool: OwnedHandle<vk::CommandPool, Device>,
    target: Option<VulkanTarget>,
    upscale_filter: vk::Filter,
    downscale_filter: vk::Filter,
    render_setups: HashMap<FormatMapping, RenderSetup>,
    submitted_frames: (mpsc::Sender<SubmittedFrame>, mpsc::Receiver<SubmittedFrame>),
    pending_frames: LinkedList<SubmittedFrame>,
}

#[doc(hidden)]
#[macro_export]
macro_rules! vk_call {
    ($dev:expr, $fn_name:ident ($($arg:expr),+ $(,)?)) => {
        unsafe {
            $dev.$fn_name($($arg),+)
                .vk(concat!("vk_", stringify!($create_fn)))
        }
    };
    ($e:expr, $fname:expr) => {
        unsafe { $e }.vk($fname)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! vk_create_guarded {
    ($dev:expr, $create_fn:ident ($($create_expr:expr),+), $destroy_fn:ident ($($destroy_expr:expr),+)) => {
        unsafe { $dev.$create_fn($($create_expr),+) }
            .vk(concat!("vk_", stringify!($create_fn)))
            .map(|v| scopeguard::guard(v, |v| unsafe {
                $dev.$destroy_fn(v, $($destroy_expr),+);
            }))
    };
}

impl VulkanRenderer {
    /// Creates a new [`VulkanRenderer`] from this [`PhysicalDevice`]. The renderer is initially
    /// bound to *no target*, meaning calling [`Renderer::render`] without first binding to a valid
    /// target is an invalid operation
    ///
    /// Will look up supported formatting information inline to cache for future reference.
    ///
    /// # Safety/Assumptions
    ///
    /// The [`PhysicalDevice`] must be on an api version greater than or equal to Vulkan 1.1
    pub fn new(phd: &PhysicalDevice) -> Result<Self, Error<'static>> {
        let mut queues = QueueFamilies::try_from(phd)?;
        let priorities = [0.0];
        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queues.graphics.index as u32)
            .queue_priorities(&priorities)
            .build()];
        let create_info = vk::DeviceCreateInfo::builder()
            .enabled_extension_names(Self::required_extensions(phd))
            .queue_create_infos(&queue_info);
        let node = phd
            .render_node()
            .ok()
            .flatten()
            .or_else(|| phd.primary_node().ok().flatten());
        let instance = phd.instance().handle();
        let device = unsafe { instance.create_device(phd.handle(), &create_info, None) }.map_or_else(
            |e| {
                Err(Error::Vk {
                    context: "vkCreateDevice",
                    result: e,
                })
            },
            |d| Ok(Device(d)),
        )?;

        queues.fill_device(&device);

        let extensions = Extensions::new(instance, &device);

        let formats: HashMap<_, _> = FormatInfo::get_known(phd).map(|f| (f.drm, f)).collect();
        let dmabuf_formats = formats
            .iter()
            .flat_map(
                |(
                    _,
                    &FormatInfo {
                        drm, ref modifiers, ..
                    },
                )| {
                    modifiers.iter().map(move |v| DrmFormat {
                        code: drm,
                        modifier: DrmModifier::from(v.drm_format_modifier),
                    })
                },
            )
            .collect::<Vec<_>>();

        let memory_props = unsafe { instance.get_physical_device_memory_properties(phd.handle()) };

        let device = Arc::new(device);

        let mut pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queues.graphics.index as _)
            .build();
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .or_else(|_| {
                    pool_info.flags = vk::CommandPoolCreateFlags::empty();
                    device.create_command_pool(&pool_info, None)
                })
                .map(|v| OwnedHandle::from_arc(v, &device))
        }.vk("vkCreateCommandPool")?;

        Ok(VulkanRenderer {
            phd: phd.clone(),
            device,
            queues,
            _node: node,
            formats,
            dmabuf_formats: dmabuf_formats.into(),
            extensions,
            debug_flags: DebugFlags::empty(),
            dmabuf_cache: HashMap::new(),
            shm_images: Vec::new(),
            memory_props,
            command_pool: pool,
            target: None,
            upscale_filter: vk::Filter::LINEAR,
            downscale_filter: vk::Filter::LINEAR,
            render_setups: HashMap::new(),
            submitted_frames: mpsc::channel(),
            pending_frames: LinkedList::new(),
        })
    }

    /// List of extensions required by a [`VulkanRenderer`]
    ///
    /// This *may* be sliced at runtime based on the vulkan version. See source for comment
    /// references on where it would be sliced for given vulkan versions
    ///
    /// # See also
    ///
    /// * [`VulkanRenderer::required_extensions`]
    const EXTS: &'static [*const i8] = &[
        // Always
        vk::ExtImageDrmFormatModifierFn::name().as_ptr(),
        vk::ExtExternalMemoryDmaBufFn::name().as_ptr(),
        vk::KhrExternalMemoryFdFn::name().as_ptr(),
        vk::KhrSwapchainFn::name().as_ptr(),
        vk::KhrSwapchainMutableFormatFn::name().as_ptr(),
        // < 1.2
        vk::KhrImageFormatListFn::name().as_ptr(),
    ];
    const VULKAN_1_2_IDX: usize = 5;

    /// Gets a list of extensions required to run a [`VulkanRenderer`] on the given
    /// [`PhysicalDevice`], as a `'static` slice of [`CStr`]-style pointers (i.e. null-terminated
    /// strings.
    ///
    /// Uses [`PhysicalDevice::api_version`] to determine what is necessary to enable
    /// given the currently-in-use API version
    pub fn required_extensions(phd: &PhysicalDevice) -> &'static [*const i8] {
        use std::ffi::CStr;
        let v = phd.api_version();
        assert_eq!(unsafe {
            CStr::from_ptr(Self::EXTS[Self::VULKAN_1_2_IDX])
        }, vk::KhrImageFormatListFn::name());
        if v < Version::VERSION_1_1 {
            panic!("unsupported vulkan api version: {:?}", v);
        } else if v >= Version::VERSION_1_2 {
            &Self::EXTS[..Self::VULKAN_1_2_IDX]
        } else {
            Self::EXTS
        }
    }

    #[inline(always)]
    pub(crate) fn instance(&self) -> &ash::Instance {
        self.phd.instance().handle()
    }

    #[inline(always)]
    pub(crate) fn device(&self) -> &ash::Device {
        &self.device
    }

    #[inline(always)]
    fn physical_device(&self) -> vk::PhysicalDevice {
        self.phd.handle()
    }

    fn format_for_drm(&self, format: &DrmFormat) -> Option<Format> {
        let &DrmFormat { code, modifier } = format;
        self.formats.get(&code).and_then(|info| {
            info.modifiers
                .iter()
                .find(|v| modifier == v.drm_format_modifier)
                .map(|mod_props| Format {
                    vk: info.vk,
                    drm: code,
                    modifier: Some(*mod_props),
                })
        })
    }

    fn find_memory_type(
        &self,
        type_mask: u32,
        props: vk::MemoryPropertyFlags,
    ) -> Option<(usize, &vk::MemoryType)> {
        self.memory_props
            .memory_types
            .iter()
            .enumerate()
            .filter(|&(idx, ..)| (type_mask & (0b1 << idx)) != 0)
            .find(|(_idx, ty)| (ty.property_flags & props) == props)
    }

    fn ensure_render_setup(
        &mut self,
        format: FormatMapping,
    ) -> Result<(), Error<'static>> {
        use std::collections::hash_map::Entry;
        let device = self.device.clone();
        if let Entry::Vacant(e) = self.render_setups.entry(format) {
            let setup = RenderSetup::new(device, format, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR)?;
            e.insert(setup);
        }
        Ok(())
    }

    /// Receives all [`SubmittedFrames`] that are pending on the receiver.
    ///
    /// the iterator will return all [`SubmittedFrame`]s that are *done*, but put the others into
    /// `pending_frames`
    fn receive_pending(&mut self) -> impl Iterator<Item=SubmittedFrame> + '_ {
        let (_, rx) = &mut self.submitted_frames;
        rx.try_iter().filter_map(|v| if v.is_done() {
            Some(v)
        } else {
            self.pending_frames.push_back(v);
            None
        })
    }

    fn cleanup_pending_threshold(&mut self, cleanup_threshold: usize) -> Option<SubmittedFrame> {
        trait CheckPendingExt {
            fn check_pending(&mut self) -> CheckPending<'_>;
        }
        impl CheckPendingExt for VulkanRenderer {
            #[inline(always)]
            fn check_pending(&mut self) -> CheckPending<'_> {
                CheckPending(self.pending_frames.cursor_front_mut())
            }
        }
        let received = self.receive_pending().last();
        let found_finished = if self.pending_frames.len() > cleanup_threshold {
            self.check_pending().last()
        } else {
            self.check_pending().next()
        };
        found_finished.or(received)
    }

    #[tracing::instrument(skip(self), name = "vulkan_renderer_cleanup")]
    #[profiling::function]
    fn cleanup(&mut self) -> Option<SubmittedFrame> {
        let has_drops = || {
            self.dmabuf_cache.keys()
                .any(WeakExt::is_gone)
            || self.shm_images.iter()
                .any(WeakExt::is_gone)
        };
        if has_drops() {
            if let Err(error) = self.wait_all_pending() {
                error!(?error, "error waiting on pending frames");
            }
        }
        self.dmabuf_cache.retain(|entry, _| entry.upgrade().is_some());
        self.shm_images.retain(|img| img.upgrade().is_some());
        self.cleanup_pending_threshold(5)
    }

    fn all_pending(&mut self) -> impl Iterator<Item=SubmittedFrame> + '_ {
        let pending = mem::take(&mut self.pending_frames);
        let (_, rx) = &self.submitted_frames;
        rx.try_iter().chain(pending)
    }

    fn wait_all_pending(&mut self) -> Result<(), Error<'static>> {
        let frames = self.all_pending()
            .collect::<Vec<_>>();
        let mut fences = Vec::with_capacity(frames.len());
        fences.extend(frames.iter().map(SubmittedFrame::fence));
        if !fences.is_empty() {
            vk_call!(self.device(), wait_for_fences(fences.as_slice(), true, u64::MAX))?;
        }
        Ok(())
    }

    /// Gets a list of all images that need to be transitioned to a correct layout
    /// (`desired_layout`) before rendering
    fn images_needing_transition(&self, desired_layout: vk::ImageLayout) -> impl Iterator<Item=VulkanImage> + '_ {
        let dmabuf_it = self.dmabuf_cache.iter()
            .filter_map(move |(_, image)| if image.0.layout.get() != desired_layout {
                Some(image.clone())
            } else {
                None
            });
        let shm_it = self.shm_images.iter()
            .filter_map(|img| img.upgrade())
            .filter(move |img| img.layout.get() != desired_layout)
            .map(VulkanImage);
        dmabuf_it
            .chain(shm_it)
    }

    fn cmd_transition_images(&self, command_buffer: vk::CommandBuffer, desired_layout: vk::ImageLayout) -> bool {
        let mut transition_it = self.images_needing_transition(desired_layout)
            .peekable();
        let ret = transition_it.has_next();
        if ret {
            // let graphics_idx = self.queues.graphics.idx();
            let barriers = transition_it.map(|image| {
                let old_layout = image.0.layout.get();
                image.0.layout.set(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(old_layout)
                    .new_layout(desired_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.0.image)
                    .subresource_range(COLOR_SINGLE_LAYER)
                    .build()
            }).collect::<Vec<_>>();
            unsafe {
                self.device().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[], &[], barriers.as_slice()
                );
            }
        }
        ret
    }
}

// impl Drop for VulkanRenderer {
//     #[tracing::instrument(skip(self), name = "vulkan_renderer_destroy")]
//     fn drop(&mut self) {
//         if let Err(error) = self.wait_all_pending() {
//             error!(?error, "error while waiting on pending frames to finish");
//         }
//     }
// }

/// [`Iterator`] implementation for iterating over [`VulkanRenderer::pending_frames`]
#[repr(transparent)]
struct CheckPending<'a>(linked_list::CursorMut<'a, SubmittedFrame>);

impl<'a> Iterator for CheckPending<'a> {
    type Item = SubmittedFrame;
    fn next(&mut self) -> Option<Self::Item> {
        while self.0.current().is_some() {
            if self.0.current().filter(|v| v.is_done()).is_some() {
                return self.0.remove_current();
            } else {
                self.0.move_next();
            }
        }
        None
    }
}

macro_rules! mocked {
    () => {
        warn!("mocked: {}", fn_name!());
    };
}

const fn filter_to_vk(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Linear => vk::Filter::LINEAR,
        TextureFilter::Nearest => vk::Filter::NEAREST,
    }
}

impl Renderer for VulkanRenderer {
    type Error = Error<'static>;
    type TextureId = VulkanImage;
    type Frame<'frame> = VulkanFrame<'frame>;

    fn id(&self) -> usize {
        use ash::vk::Handle;
        let device: &ash::Device = &self.device;
        device.handle().as_raw() as usize
    }
    fn downscale_filter(&mut self, f: TextureFilter) -> Result<(), Self::Error> {
        self.downscale_filter = filter_to_vk(f);
        Ok(())
    }
    fn upscale_filter(&mut self, f: TextureFilter) -> Result<(), Self::Error> {
        self.upscale_filter = filter_to_vk(f);
        Ok(())
    }
    fn set_debug_flags(&mut self, flags: DebugFlags) {
        warn!("debug flags currently ignored: {:?}", flags);
        self.debug_flags = flags;
    }
    fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }
    #[tracing::instrument(skip(self))]
    fn render(
        &mut self,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<Self::Frame<'_>, Self::Error> {
        frame::render_internal(self, output_size, dst_transform)
    }
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.device.wait_fence_vk(sync)
    }
}

#[derive(Debug)]
enum VulkanTarget {
    Surface(Rc<crate::backend::vulkan::Surface>, Swapchain),
}

impl PartialEq<crate::backend::vulkan::Surface> for VulkanTarget {
    fn eq(&self, rhs: &crate::backend::vulkan::Surface) -> bool {
        match self {
            VulkanTarget::Surface(surface, swapchain) => {
                surface.handle() == rhs.handle() && swapchain.extent() == &rhs.extent()
            },
        }
    }
}

impl super::Bind<Rc<crate::backend::vulkan::Surface>> for VulkanRenderer {
    #[tracing::instrument(skip(self))]
    fn bind(
        &mut self,
        target: Rc<crate::backend::vulkan::Surface>,
    ) -> Result<(), <Self as Renderer>::Error> {
        match &self.target {
            Some(tgt) if tgt == &*target => Ok(()),
            _ => {
                use swapchain::SupportDetails;
                debug!("waiting on pending/in-flight frames to complete");
                self.wait_all_pending()?;

                let swapchain_support = SupportDetails::with_surface(
                    self.phd.handle(),
                    target.handle(),
                    target.extension(),
                )?;
                let swapchain::SurfaceFormatInfo { format, color_space } = swapchain_support.choose_format()
                    .ok_or(Error::SwapchainFormat)?;

                self.ensure_render_setup(format)?;
                let render_setup = self.render_setups.get(&format)
                    .ok_or(Error::RendererSetup(format))?;

                if matches!(
                    &self.target,
                    Some(VulkanTarget::Surface(_, swapchain)) if swapchain.surface() == target.handle()
                ) {
                    self.target = None;
                }

                let swapchain = Swapchain::with_surface(
                    &*self,
                    swapchain::SurfaceInfo::new(
                        target.handle(),
                        target.extension(),
                        color_space,
                        &swapchain_support
                    ),
                    &target.extent(),
                    render_setup,
                )?;
                let surface = target.handle();
                debug!(?swapchain, ?surface, "created swapchain");
                self.target = Some(VulkanTarget::Surface(target, swapchain));
                Ok(())
            },
        }
    }
}

impl super::Unbind for VulkanRenderer {
    fn unbind(&mut self) -> Result<(), <Self as Renderer>::Error> {
        self.wait_all_pending()?;
        self.target = None;
        Ok(())
    }
}

/// Maximum number of planes supported for imported [`Dmabuf`]s
///
/// Used to statically-size arrays relative to planes to avoid extra allocations being required on
/// each call to [`ImportDma::import_dmabuf`] for [`VulkanRenderer`]
const MAX_PLANES: usize = 4;

fn imported_usage_flags() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::SAMPLED
    | vk::ImageUsageFlags::TRANSFER_SRC
    | vk::ImageUsageFlags::TRANSFER_DST
}

impl ImportDma for VulkanRenderer {
    #[tracing::instrument(skip(self, _damage))]
    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<<Self as Renderer>::TextureId, <Self as Renderer>::Error> {
        use dmabuf::*;
        const H_TYPE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        const ALL_PLANE_ASPECTS: [vk::ImageAspectFlags; MAX_PLANES] = [
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
        ];

        if let Some(existing) = self
            .dmabuf_cache
            .get_key_value(&dmabuf.weak())
            .and_then(|(weak, img)| {
                weak.upgrade().filter(|buf| buf == dmabuf)?;
                Some(img.clone())
            })
        {
            return Ok(existing);
        }

        let fmt = self
            .format_for_drm(&dmabuf.format())
            .ok_or(Error::UnknownFormat(dmabuf.format()))?;
        debug!(?fmt, ?dmabuf, "found format for dmabuf");

        let disjoint = dmabuf.is_disjoint()?;
        let image_create_flags = if disjoint {
            vk::ImageCreateFlags::DISJOINT
        } else {
            vk::ImageCreateFlags::empty()
        };
        let mut external_info = vk::PhysicalDeviceExternalImageFormatInfo::builder().handle_type(H_TYPE);
        let mut info = vk::PhysicalDeviceImageFormatInfo2::builder()
            .format(fmt.vk)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(imported_usage_flags())
            .flags(image_create_flags)
            .push_next(&mut external_info);
        let mut drm_info: vk::PhysicalDeviceImageDrmFormatModifierInfoEXT;
        let queue_indices = [self.queues.graphics.index as u32];
        if let Some(mod_info) = fmt.modifier.as_ref() {
            drm_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::builder()
                .drm_format_modifier(mod_info.drm_format_modifier)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&queue_indices)
                .build();
            info = info.push_next(&mut drm_info);
        }
        let mut external_fmt_props = vk::ExternalImageFormatProperties::default();
        let mut image_fmt = vk::ImageFormatProperties2::builder().push_next(&mut external_fmt_props);
        let _image_fmt = unsafe {
            self.instance()
                .get_physical_device_image_format_properties2(self.phd.handle(), &info, &mut image_fmt)
                .vk("vkGetPhysicalDeviceImageFormatProperties2")?;
            let image_fmt_props = image_fmt.image_format_properties;
            trace!(
                "vkGetPhysicalDeviceImageFormatProperties2:\n{:#?}",
                &image_fmt_props
            );
            image_fmt_props
        };

        let mut drm_info: vk::ImageDrmFormatModifierExplicitCreateInfoEXTBuilder<'_>;
        let mut plane_layouts: Vec<vk::SubresourceLayout>;
        let mut external_info = vk::ExternalMemoryImageCreateInfo::builder().handle_types(H_TYPE);
        let extent @ vk::Extent3D { width, height, .. } = dmabuf.extent_3d();
        let mut info = vk::ImageCreateInfo::builder()
            .push_next(&mut external_info)
            .flags(image_create_flags)
            .image_type(vk::ImageType::TYPE_2D)
            .format(fmt.vk)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(imported_usage_flags())
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(queue_indices.as_slice())
            .initial_layout(vk::ImageLayout::UNDEFINED);
        if let Some(mod_info) = fmt.modifier.as_ref() {
            plane_layouts = Vec::with_capacity(dmabuf.num_planes());
            let it = dmabuf
                .offsets()
                .zip(dmabuf.strides())
                .map(|(offset, stride)| vk::SubresourceLayout {
                    offset: offset as _,
                    size: 0,
                    row_pitch: stride as _,
                    array_pitch: 0,
                    depth_pitch: 0,
                });
            plane_layouts.extend(it);
            drm_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::builder()
                .drm_format_modifier(mod_info.drm_format_modifier)
                .plane_layouts(plane_layouts.as_slice());
            info = info.push_next(&mut drm_info);
        }
        let image = unsafe { self.device.create_image(&info, None) }.map_or_else(
            |e| {
                Err(Error::Vk {
                    context: "vkCreateImage",
                    result: e,
                })
            },
            |img| {
                Ok(scopeguard::guard(img, |img| unsafe {
                    self.device.destroy_image(img, None);
                }))
            },
        )?;
        trace!("imported VkImage: {:?}", &image);

        let memory_plane_props = self
            .extensions
            .external_memory_fd
            .dmabuf_memory_properties(dmabuf)
            .map_err(|e| Error::Vk {
                context: "get dmabuf fd memory properties",
                result: e,
            })?;
        let mem_count = if disjoint {
            std::cmp::min(memory_plane_props.len(), ALL_PLANE_ASPECTS.len())
        } else {
            1usize
        };
        let mut bind_plane_infos = [vk::BindImagePlaneMemoryInfo::default(); MAX_PLANES];
        let mut bind_infos = [vk::BindImageMemoryInfo::default(); MAX_PLANES];
        let mut memories = scopeguard::guard([vk::DeviceMemory::null(); MAX_PLANES], |memories| unsafe {
            for m in memories {
                if !m.is_null() {
                    self.device.free_memory(m, None);
                }
            }
        });
        memory_plane_props
            .into_iter()
            .enumerate()
            .zip(memories.iter_mut())
            .take(mem_count)
            .try_for_each(|((idx, (fd, props)), dm)| -> Result<_, Error<'static>> {
                let mem_reqs = self
                    .device
                    .get_memory_requirements(*image, ALL_PLANE_ASPECTS[idx], disjoint)
                    .memory_requirements;
                let mem_type_bits = props.memory_type_bits & mem_reqs.memory_type_bits;
                let (mem_idx, ..) = self
                    .find_memory_type(mem_type_bits, vk::MemoryPropertyFlags::empty())
                    .ok_or(DmabufError::NoMemoryType(mem_type_bits))?;
                let fd = fd
                    .try_clone_to_owned()
                    .map_err(|e| Error::from(DmabufError::Io(e)))?;
                let mut import_info = vk::ImportMemoryFdInfoKHR::builder()
                    .handle_type(H_TYPE)
                    .fd(fd.as_raw_fd());
                let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::builder().image(*image);
                let create_info = vk::MemoryAllocateInfo::builder()
                    .allocation_size(mem_reqs.size)
                    .memory_type_index(mem_idx as _)
                    .push_next(&mut dedicated_info)
                    .push_next(&mut import_info);
                *dm = vk_call!(
                    self.device.allocate_memory(&create_info, None),
                    "vkAllocateMemory"
                )?;
                trace!("vkAllocateMemory:\n{:#?} -> {:?}", &create_info.build(), dm);

                // this fd will be closed by vulkan once the image is free'd
                fd.into_raw_fd();
                bind_plane_infos[idx] = vk::BindImagePlaneMemoryInfo::builder()
                    .plane_aspect(ALL_PLANE_ASPECTS[idx])
                    .build();
                bind_infos[idx] = vk::BindImageMemoryInfo::builder()
                    .image(*image)
                    .memory(*dm)
                    .memory_offset(0)
                    .build();
                if disjoint {
                    let p = &bind_plane_infos[idx] as *const vk::BindImagePlaneMemoryInfo;
                    bind_infos[idx].p_next = p.cast();
                }
                Ok(())
            })?;
        let mems = &memories[..mem_count];
        trace!(
            "import_dmabuf: imported device memories[{}]:\n{:#?}",
            mems.len(),
            mems
        );

        vk_call!(
            self.device.bind_image_memory2(&bind_infos[..mems.len()]),
            "vkBindImageMemory2"
        )?;

        let image = VulkanImage::new(InnerImage {
            image: ScopeGuard::into_inner(image),
            memories: ScopeGuard::into_inner(memories),
            buffer: None,
            dimensions: (width, height),
            format: fmt,
            views: RefCell::new(Vec::new()),
            layout: Cell::new(vk::ImageLayout::UNDEFINED),
            device: Arc::downgrade(&self.device),
        });
        self.dmabuf_cache.insert(dmabuf.weak(), image.clone());
        Ok(image)
    }

    fn dmabuf_formats(&self) -> Box<dyn Iterator<Item = DrmFormat>> {
        let it = DerefSliceIter::new(self.dmabuf_formats.clone());
            // .filter(|&DrmFormat { code, .. }| !matches!(
            //     code,
            //     Fourcc::Argb8888
            //     | Fourcc::Abgr8888,
            // ));
        Box::new(it) as Box<_>
    }

    fn has_dmabuf_format(&self, format: DrmFormat) -> bool {
        self.format_for_drm(&format).is_some()
    }
}

impl ImportDmaWl for VulkanRenderer {}

fn mem_formats() -> impl Iterator<Item = (Fourcc, vk::Format)> {
    use crate::backend::allocator::vulkan::format;
    fn is_match(f: vk::Format) -> bool {
        matches!(
            f,
            vk::Format::B8G8R8A8_SRGB
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::B8G8R8A8_UNORM
            | vk::Format::R8G8B8A8_UNORM
            | vk::Format::B8G8R8_SRGB
            | vk::Format::R8G8B8_SRGB,
        )
    }
    FORMAT_MAPPINGS.iter()
        .flat_map(|&(drm, fm)| {
            fm.srgb()
                .filter(|&f| is_match(f))
                .map(|f| (drm, f))
                .or_else(|| if is_match(fm.format) {
                    Some((drm, fm.format))
                } else {
                    None
                })
        })
}

impl ImportMem for VulkanRenderer {
    fn import_memory(
        &mut self,
        _data: &[u8],
        _format: Fourcc,
        _size: Size<i32, BufferCoord>,
        _flipped: bool,
    ) -> Result<<Self as Renderer>::TextureId, <Self as Renderer>::Error> {
        todo!()
    }

    fn update_memory(
        &mut self,
        _texture: &<Self as Renderer>::TextureId,
        _data: &[u8],
        _region: Rectangle<i32, BufferCoord>,
    ) -> Result<(), <Self as Renderer>::Error> {
        todo!()
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        LOG_MEM_FORMATS.call_once(|| {
            let shm_formats: Box<[(Fourcc, vk::Format)]> = mem_formats()
                .collect::<Vec<_>>()
                .into();
            debug!(shm_formats = ?&*shm_formats, "supported shm formats");
        });
        Box::new(mem_formats().map(|(fmt, ..)| fmt))
    }
}

use std::sync::Once;
static LOG_MEM_FORMATS: std::sync::Once = std::sync::Once::new();

type ShmCache = HashMap<usize, Arc<InnerImage>>;

/// Helper trait for [`ImportMemWl`] `impl` for [`VulkanRenderer`]
trait SurfaceExtVulkan {
    fn cached_image<R: Renderer>(&self, r: &R, data: &BufferData) -> Option<Arc<InnerImage>>;
}

impl SurfaceExtVulkan for compositor::SurfaceData {
    fn cached_image<R: Renderer>(&self, r: &R, data: &BufferData) -> Option<Arc<InnerImage>> {
        let &BufferData { width, height, .. } = data;
        self
            .data_map
            .insert_if_missing(|| Rc::new(RefCell::new(ShmCache::new())));
        self
            .data_map
            .get::<Rc<RefCell<ShmCache>>>()
            .unwrap()
            .borrow()
            .get(&r.id())
            .cloned()
            .filter(|tex| tex.dimensions == (width as _, height as _))
    }
}

impl ImportMemWl for VulkanRenderer {
    #[tracing::instrument(skip(self, _damage))]
    fn import_shm_buffer(
        &mut self,
        buffer: &wayland_server::protocol::wl_buffer::WlBuffer,
        surface: Option<&compositor::SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<<Self as Renderer>::TextureId, <Self as Renderer>::Error> {
        let ret = with_buffer_contents(buffer, |ptr, len, data| -> Result<_, <Self as Renderer>::Error> {
            let BufferData { offset, width, height, format: shm_format, stride } = data;

            let bytes_per_pixel = stride / width;

            debug!(len, buffer_data = ?data, "got buffer data");

            // try cache first
            let image = if let Some(cached) = surface.and_then(|v| v.cached_image(&*self, &data)) {
                cached.clone()
            } else {
                // otherwise create a new image
                let device = self.device();
                let fmt = Format::try_from(shm_format)?;
                debug!(?fmt, "found format for shm buffer");
                let queue_family_indices = &[self.queues.graphics.index as u32];
                let fmt_mapping = FORMAT_MAPPINGS.iter()
                    .find_map(|&(drm_fmt, fm)| if drm_fmt == fmt.drm {
                        Some(fm)
                    } else {
                        None
                    })
                    .unwrap_or_else(|| {
                        warn!(?fmt, "no format mapping found for shm input format");
                        FormatMapping::from(fmt.vk)
                    });
                assert_eq!(fmt_mapping.format, fmt.vk);
                let view_formats = [
                    fmt.vk,
                    fmt_mapping.srgb().unwrap_or(vk::Format::default()),
                ];
                let mut view_formats_list = vk::ImageFormatListCreateInfoKHR::builder()
                    .view_formats(&view_formats);
                let mut info = vk::ImageCreateInfo::builder()
                    .flags(vk::ImageCreateFlags::empty())
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(fmt.vk)
                    .extent(vk::Extent3D {
                        width: width as _,
                        height: height as _,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::LINEAR)
                    .usage(imported_usage_flags())
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .queue_family_indices(queue_family_indices)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                if view_formats[1] != vk::Format::default() {
                    info = info
                        .push_next(&mut view_formats_list)
                        .flags(vk::ImageCreateFlags::MUTABLE_FORMAT);
                }
                let image = vk_create_guarded!(device, create_image(&info, None), destroy_image(None))?;
                let reqs = unsafe { device.get_image_memory_requirements(*image) };
                let (mem_idx, ..) = [
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::empty(),
                ].into_iter().find_map(|props| {
                    self.find_memory_type(reqs.memory_type_bits, props)
                }).ok_or(Error::NoMemoryType {
                    bits: reqs.memory_type_bits,
                    flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                })?;
                let info = vk::MemoryAllocateInfo::builder()
                    .allocation_size(reqs.size)
                    .memory_type_index(mem_idx as _)
                    .build();
                let memory = vk_create_guarded!(device, allocate_memory(&info, None), free_memory(None))?;
                vk_call!(device, bind_image_memory(*image, *memory, 0))?;

                let buf_size = ((width * height) as usize) * bytes_per_pixel as usize;
                debug_assert!(offset as vk::DeviceSize + buf_size as vk::DeviceSize <= len as vk::DeviceSize);

                let buffer = StagingBuffer::new(self, buf_size as vk::DeviceSize, vk::SharingMode::EXCLUSIVE)?;
                let mut ret = InnerImage {
                    image: ScopeGuard::into_inner(image),
                    memories: [vk::DeviceMemory::null(); MAX_PLANES],
                    buffer: Some(buffer),
                    dimensions: (width as _, height as _),
                    format: fmt,
                    views: RefCell::new(Vec::new()),
                    layout: Cell::new(vk::ImageLayout::UNDEFINED),
                    device: Arc::downgrade(&self.device),
                };
                ret.memories[0] = ScopeGuard::into_inner(memory);
                Arc::new(ret)
            };

            // update data in staging buffer
            let b_data = unsafe {
                std::slice::from_raw_parts(ptr, len)
            };
            // TODO try not repeat work
            let buffer = image.buffer.as_ref()
                .expect("shm image must have a buffer");
            let b_data = &b_data[offset as usize..buffer.len()];
            buffer.upload(b_data, 0)?;

            // submit a copy from the staging buffer to the main image
            let cb = self.device.create_single_command_buffer(
                *self.command_pool,
                vk::CommandBufferLevel::PRIMARY,
            ).map(|v| scopeguard::guard(v, |v| unsafe {
                self.device.free_command_buffers(*self.command_pool, &[v]);
            }))?;
            let (w, h) = image.dimensions;
            buffer::record_copy_buffer_image(
                self.device(),
                &self.queues.graphics,
                *cb, buffer,
                &image, [w, h], 0,
            )?;
            let info = vk::FenceCreateInfo::builder()
                .flags(vk::FenceCreateFlags::empty());
            let fence = vk_create_guarded!(self.device(), create_fence(&info, None), destroy_fence(None))?;
            let cbs = &[*cb];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&[])
                .wait_dst_stage_mask(&[])
                .command_buffers(cbs)
                .signal_semaphores(&[]);

            // TODO: make it so we don't have to block here by using semaphores
            //
            // or maybe moving these commands into the render submission?
            vk_call!(self.device(), queue_submit(self.queues.graphics.handle, &[submit_info.build()], *fence))?;
            image.layout.set(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            vk_call!(self.device(), wait_for_fences(&[*fence], true, u64::MAX))?;

            // self.shm_images.push(Arc::downgrade(&image));

            if let Some(surface) = surface.as_ref() {
                let mut cache = surface
                    .data_map
                    .get::<Rc<RefCell<ShmCache>>>()
                    .unwrap()
                    .borrow_mut();
                cache.insert(self.id(), image.clone());
            }

            Ok(VulkanImage(image))
        }).map_err(Error::BufferAccess)?;
        match &ret {
            Ok(VulkanImage(ref inner)) => {
                // if let Ok(VulkanImage(ref inner)) = &ret {
                // }
                self.shm_images.push(Arc::downgrade(inner));
            },
            Err(error) => {
                error!(?error, "failed");
            },
        }
        debug!(
            ?ret,
            shm_images = ?self.shm_images.as_slice(),
            n_shm_images = self.shm_images.len(),
            "finished import"
        );
        // if let Ok(image) = &ret {
        //     self.shm_images.push(image.clone());
        // }
        ret
    }
}

#[cfg(all(
    feature = "wayland_frontend",
    feature = "backend_egl",
    feature = "use_system_lib"
))]
impl super::ImportEgl for VulkanRenderer {
    fn bind_wl_display(&mut self, _display: &wayland_server::DisplayHandle) -> Result<(), crate::backend::egl::Error> {
        mocked!();
        Ok(())
    }

    fn unbind_wl_display(&mut self) {
        mocked!();
    }

    fn egl_reader(&self) -> Option<&crate::backend::egl::display::EGLBufferReader> {
        mocked!();
        None
    }

    fn import_egl_buffer(
        &mut self,
        _buffer: &wayland_server::protocol::wl_buffer::WlBuffer,
        _surface: Option<&crate::wayland::compositor::SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<<Self as Renderer>::TextureId, <Self as Renderer>::Error> {
        // Err(Error::Unimplemented(fn_name!()))
        unimplemented!()
    }
}

/// [`Texture`] implementation for a [`VulkanRenderer`]
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct VulkanImage(Arc<InnerImage>);

type WeakVulkanImage = Weak<InnerImage>;

impl VulkanImage {
    #[inline(always)]
    fn new(img: InnerImage) -> Self {
        VulkanImage(Arc::new(img))
    }
}

impl Texture for VulkanImage {
    fn width(&self) -> u32 {
        self.0.dimensions.0
    }
    fn height(&self) -> u32 {
        self.0.dimensions.1
    }
    fn format(&self) -> Option<Fourcc> {
        Some(self.0.format.drm)
    }
}

#[derive(Debug)]
struct InnerImage {
    image: vk::Image,
    memories: [vk::DeviceMemory; MAX_PLANES],
    buffer: Option<StagingBuffer>,
    dimensions: (u32, u32),
    format: Format,
    views: RefCell<Vec<(vk::DescriptorSetLayout, InnerImageView)>>,
    layout: Cell<vk::ImageLayout>,
    device: Weak<Device>,
}

impl InnerImage {
    #[inline(always)]
    fn memories(&self) -> impl Iterator<Item=&'_ vk::DeviceMemory> {
        self.memories.iter()
            .filter(|&&mem| mem != vk::DeviceMemory::null())
    }

    #[inline]
    fn has_alpha(&self) -> bool {
        use crate::backend::allocator::vulkan::format::FormatExt;
        self.format.vk.has_alpha()
    }

    /// TODO: honestly we *probably* know when this is going to happen, so wouldn't have to key off
    /// of it and have mutable state within [`InnerImage`].
    ///
    /// Fixing this would also avoid the dynamic allocation of the [`Vec`] for the
    /// [`InnerImageView`]s
    fn get_or_create_view(&self, layout: &render_pass::PipelineLayout) -> Result<impl Deref<Target=InnerImageView> + '_, Error<'static>> {
        use std::cell::Ref;
        debug_assert_ne!(layout.ds_layout, vk::DescriptorSetLayout::null());
        {
            let views = self.views.borrow();
            let get_view = |key: vk::DescriptorSetLayout| Ref::filter_map(views, |views| {
                views.iter().find_map(|&(layout, ref view)| if layout == key {
                    Some(view)
                } else {
                    None
                })
            });
            if let Ok(v) = get_view(layout.ds_layout) {
                return Ok(v);
            }
        }
        {
            let mut views = self.views.borrow_mut();
            // TODO: fuck this '4' cant be hard-coded
            let view = InnerImageView::new(self, layout, 6)?;
            views.push((layout.ds_layout, view));
        }
        Ok(Ref::map(self.views.borrow(), |views| &views[views.len()-1].1))
    }
}

impl Drop for InnerImage {
    fn drop(&mut self) {
        let Some(device) = self.device.upgrade() else {
            error!("device destroyed before image: {:?}", self.image);
            return;
        };
        unsafe {
            // if !self.command_buffers[0].is_null() {
            //     let end_idx = self.command_buffers.iter()
            //         .position(|&v| v.is_null())
            //         .unwrap_or(self.command_buffers.len());
            //     device.free_command_buffers(self.command_pool_ref, &self.command_buffers[..end_idx]);
            // }
            let mut views = self.views.borrow_mut();
            for (_, view) in &mut *views {
                view.destroy(&device);
            }
            device.destroy_image(self.image, None);
            for &mem in self.memories() {
                device.free_memory(mem, None);
            }
        }
    }
}

struct Extensions {
    _image_format_modifier: ImageDrmFormatModifier,
    external_memory_fd: ExternalMemoryFd,
}

impl fmt::Debug for Extensions {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Extensions")
    }
}

impl Extensions {
    #[inline(always)]
    pub fn new(i: &ash::Instance, d: &ash::Device) -> Self {
        Extensions {
            _image_format_modifier: ImageDrmFormatModifier::new(i, d),
            external_memory_fd: ExternalMemoryFd::new(i, d),
        }
    }
}

#[repr(transparent)]
pub(crate) struct Device(ash::Device);
impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.0.destroy_device(None);
        }
    }
}
impl Deref for Device {
    type Target = ash::Device;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Device {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Device").field(&self.0.handle()).finish()
    }
}

/// [`Error`](std::error::Error) enum representing errors that can occur working with a
/// [`VulkanRenderer`]
#[derive(Debug, thiserror::Error)]
pub enum Error<'a> {
    /// Occurs when the [`PhysicalDevice`] used to create the [`VulkanRenderer`] does not support a
    /// valid `GRAPHICS` queue family
    #[error("physical device had no graphics queue family")]
    NoGraphicsQueue,
    /// Device didn't have a supported presentation queue
    #[error("physical device had no presentation queue for surface")]
    NoPresentQueue,
    /// Wrapper for [`vk::Result`], including `context` as the name of the vulkan function that
    /// triggered the wrapped error
    #[error("{context}: {result}")]
    Vk {
        /// contextual information about what caused the given error.
        ///
        /// This is usually the name of the vulkan function from which the error originated
        context: &'a str,
        /// Inner [`vk::Result`] value returned by the function named in `context`
        result: vk::Result,
    },
    /// Simple wrapper for [`dmabuf::DmabufError`]
    #[error("error importing dmabuf: {0}")]
    Dmabuf(#[from] dmabuf::DmabufError),
    /// Occurs when [`ImportDma::import_dmabuf`] is called with an unknown format
    #[error("failed to convert import format: {0:?}")]
    UnknownFormat(DrmFormat),
    /// Occurs when [`ImportMem`] is called with an unsupported format
    #[error("unsupported memory import format: {0:?}")]
    /// Desired import [`Fourcc`] drm format not supported
    UnknownMemoryFormat(Fourcc),
    /// user supplied an unsupported [`wl_shm::Format`] value to [`ImportMemWl::import_shm_buffer`]
    #[error("unsupported wl_shm format: {0:?}")]
    UnknownShmFormat(wl_shm::Format),
    /// device did not support required memory parameters
    #[error("no device memory type with bits: {bits:b} and flags {flags:?}")]
    NoMemoryType {
        /// `memory_type_bits` from EX [`vk::ImageMemoryRequirements`]
        bits: u32,
        /// flags that the memory must support that were not found
        flags: vk::MemoryPropertyFlags,
    },
    /// Occurs when [`Renderer::render`] is called without a target bound for the renderer
    #[error("no render target bound")]
    NoTarget,
    /// Occurs if a new rendering frame is started, but all of the available command buffers are
    /// already busy
    #[error("all command buffers already busy")]
    AllCommandBuffersBusy,
    /// swapchain support not found for surface
    #[error("swapchain support not found for surface")]
    SwapchainSupport,
    /// could not find compatible swapchain format
    #[error("could not find compatible swapchain format")]
    SwapchainFormat,
    /// Feature named is unimplemented
    #[error("unimplemented: {0}")]
    Unimplemented(&'a str),
    /// ?
    #[error("no renderer setup found for format: {0:?}")]
    RendererSetup(FormatMapping),
    /// an interrupt occured while waiting on a [`SyncPoint`]
    #[error("waiting for fence interrupted: {0}")]
    Interrupted(#[from] crate::backend::renderer::sync::Interrupted),
    /// Returned by [`ImportMemWl::import_shm_buffer`] when the import causes an invalid access to
    /// the buffer's memory
    #[error(transparent)]
    BufferAccess(#[from] crate::wayland::shm::BufferAccessError),
}

impl<'a> Error<'a> {
    /// # See also
    ///
    /// * [`ErrorExt::vk`]
    #[inline(always)]
    fn vk(s: &'a str) -> impl Fn(vk::Result) -> Self + 'a {
        move |e: vk::Result| Error::Vk {
            context: s,
            result: e,
        }
    }
}

/// Helper trait for [`VkResult`]
pub(crate) trait ErrorExt {
    type Ret;
    /// Helper function for converting [`VkResult`] values to [`Result`]s with the error as
    /// [`Error`]
    fn vk(self, ctx: &str) -> Result<Self::Ret, Error<'_>>;
}

impl<T> ErrorExt for VkResult<T> {
    type Ret = T;
    #[inline(always)]
    fn vk(self, ctx: &str) -> Result<Self::Ret, Error<'_>> {
        self.map_err(Error::vk(ctx))
    }
}

trait DeviceExt {
    fn memory_requirements(&self, image: vk::Image) -> vk::MemoryRequirements2;
    fn disjoint_memory_requirements(
        &self,
        image: vk::Image,
        plane_aspect: vk::ImageAspectFlags,
    ) -> vk::MemoryRequirements2;

    #[inline(always)]
    fn get_memory_requirements(
        &self,
        image: vk::Image,
        plane_aspect: vk::ImageAspectFlags,
        disjoint: bool,
    ) -> vk::MemoryRequirements2 {
        if disjoint {
            self.disjoint_memory_requirements(image, plane_aspect)
        } else {
            self.memory_requirements(image)
        }
    }
    fn push_constant<T: Sized>(
        &self,
        cb: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        stage: vk::ShaderStageFlags,
        offset: usize,
        value: &T,
    );
    fn create_single_command_buffer(
        &self,
        pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer, Error<'static>>;
}
impl DeviceExt for ash::Device {
    fn memory_requirements(&self, image: vk::Image) -> vk::MemoryRequirements2 {
        let info = vk::ImageMemoryRequirementsInfo2::builder().image(image);
        let mut ret = vk::MemoryRequirements2::default();
        unsafe {
            self.get_image_memory_requirements2(&info, &mut ret);
        }
        ret
    }
    fn disjoint_memory_requirements(
        &self,
        image: vk::Image,
        plane_aspect: vk::ImageAspectFlags,
    ) -> vk::MemoryRequirements2 {
        let mut info_plane = vk::ImagePlaneMemoryRequirementsInfo::builder().plane_aspect(plane_aspect);
        let info = vk::ImageMemoryRequirementsInfo2::builder()
            .push_next(&mut info_plane)
            .image(image);
        let mut ret = vk::MemoryRequirements2::default();
        unsafe {
            self.get_image_memory_requirements2(&info, &mut ret);
        }
        ret
    }
    #[inline]
    fn push_constant<T: Sized>(
        &self,
        cb: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        stage: vk::ShaderStageFlags,
        offset: usize,
        value: &T,
    ) {
        unsafe {
            let raw_data = std::slice::from_raw_parts(
                (value as *const T).cast(),
                mem::size_of::<T>()
            );
            self.cmd_push_constants(
                cb, layout, stage,
                offset as _,
                raw_data
            );
        }
    }
    fn create_single_command_buffer(
        &self,
        pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer, Error<'static>> {
        use core::mem::MaybeUninit;
        let mut ret = MaybeUninit::<vk::CommandBuffer>::zeroed();
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(level)
            .command_buffer_count(1)
            .build();
        unsafe {
            let result = (self.fp_v1_0().allocate_command_buffers)(
                self.handle(),
                &alloc_info as *const _,
                ret.as_mut_ptr(),
            );
            match result {
                vk::Result::SUCCESS => Ok(ret.assume_init()),
                e => Err(Error::Vk {
                    context: "vkAllocateCommandBuffers",
                    result: e,
                }),
            }
        }
    }
}

const COLOR_SINGLE_LAYER: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

contextual_handles!(Device {
    vk::CommandPool = destroy_command_pool,
    vk::RenderPass = destroy_render_pass,
});

vulkan_handles! {
    vk::CommandBuffer,
    vk::DeviceMemory,
    vk::Semaphore,
    vk::Fence,
}
