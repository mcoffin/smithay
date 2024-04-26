use super::{ImportDma, ImportDmaWl, ImportMem, ImportMemWl};
use crate::{
    backend::{
        allocator::{
            dmabuf::{Dmabuf, WeakDmabuf},
            Buffer, Format as DrmFormat, Fourcc, Modifier as DrmModifier,
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
};
use ash::{
    extensions::{ext::ImageDrmFormatModifier, khr::ExternalMemoryFd},
    prelude::VkResult,
    vk,
};
use scopeguard::ScopeGuard;
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt,
    ops::{Deref, DerefMut},
    os::fd::{AsRawFd, IntoRawFd},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
};

#[allow(unused_imports)]
use tracing::{debug, error, trace, warn};

mod command_pool;
mod dmabuf;
mod fence;
mod render_pass;
mod swapchain;
mod util;
use util::*;
use fence::*;
use swapchain::Swapchain;
use render_pass::RenderSetup;

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
    memory_props: vk::PhysicalDeviceMemoryProperties,
    command_pool: OwnedHandle<vk::CommandPool, Device>,
    // command_buffers: Box<[vk::CommandBuffer]>,
    // command_buffer_usage: Box<[AtomicBool]>,
    target: Option<VulkanTarget>,
    upscale_filter: vk::Filter,
    downscale_filter: vk::Filter,
    render_setups: HashMap<vk::Format, RenderSetup>,
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
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
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

        // let cmd_buf_alloc_info = vk::CommandBufferAllocateInfo::builder()
        //     .command_pool(*pool)
        //     .level(vk::CommandBufferLevel::PRIMARY)
        //     .command_buffer_count(DEFAULT_COMMAND_BUFFERS as _);
        // let cmd_buffers =
        //     unsafe { device.allocate_command_buffers(&cmd_buf_alloc_info) }.vk("vkAllocateCommandBuffers")?;

        // let cmd_buffer_usage = {
        //     let mut ret = Vec::with_capacity(cmd_buffers.len());
        //     let usage_it = std::iter::repeat_with(|| AtomicBool::new(false));
        //     ret.extend(usage_it.take(cmd_buffers.len()));
        //     ret.into()
        // };

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
            memory_props,
            command_pool: pool,
            // command_buffers: cmd_buffers.into(),
            // command_buffer_usage: cmd_buffer_usage,
            target: None,
            upscale_filter: vk::Filter::LINEAR,
            downscale_filter: vk::Filter::LINEAR,
            render_setups: HashMap::new(),
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
        // < 1.2
        vk::KhrImageFormatListFn::name().as_ptr(),
    ];

    /// Gets a list of extensions required to run a [`VulkanRenderer`] on the given
    /// [`PhysicalDevice`], as a `'static` slice of [`CStr`]-style pointers (i.e. null-terminated
    /// strings.
    ///
    /// Uses [`PhysicalDevice::api_version`] to determine what is necessary to enable
    /// given the currently-in-use API version
    pub fn required_extensions(phd: &PhysicalDevice) -> &'static [*const i8] {
        let v = phd.api_version();
        if v < Version::VERSION_1_1 {
            panic!("unsupported vulkan api version: {:?}", v);
        } else if v >= Version::VERSION_1_2 {
            &Self::EXTS[..4]
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
        format: vk::Format
    ) -> Result<(), Error<'static>> {
        use std::collections::hash_map::Entry;
        let device = self.device.clone();
        if let Entry::Vacant(e) = self.render_setups.entry(format) {
            let setup = RenderSetup::new(device, format, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR)?;
            e.insert(setup);
        }
        Ok(())
    }

    // #[inline(always)]
    // fn command_buffers(&self) -> impl Iterator<Item = (vk::CommandBuffer, &AtomicBool)> {
    //     self.command_buffers
    //         .iter()
    //         .copied()
    //         .zip(&*self.command_buffer_usage)
    // }

    // /// TODO: verify this AtomicBool logic and ordering w/ [`CommandBuffer::drop`]
    // fn acquire_command_buffer(&self) -> Result<CommandBuffer<'_>, Error<'static>> {
    //     self.command_buffers()
    //         .find(|&(_buf, in_use)| {
    //             in_use
    //                 .compare_exchange(false, true, Ordering::SeqCst, Ordering::Acquire)
    //                 .is_ok()
    //         })
    //         .map(CommandBuffer::from)
    //         .ok_or(Error::AllCommandBuffersBusy)
    // }
}

macro_rules! vk_call {
    ($e:expr, $fname:expr) => {
        unsafe { $e }.vk($fname)
    };
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
    fn render(
        &mut self,
        output_size: Size<i32, Physical>,
        _dst_transform: Transform,
    ) -> Result<Self::Frame<'_>, Self::Error> {
        use std::mem::MaybeUninit;
        let target = self.target.as_ref().ok_or(Error::NoTarget)?;
        let command_buffer = unsafe {
            let mut ret = MaybeUninit::<vk::CommandBuffer>::zeroed();
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(*self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1)
                .build();
            let result = (self.device().fp_v1_0().allocate_command_buffers)(
                self.device().handle(),
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
        }?;
        let mut ret = VulkanFrame {
            renderer: self,
            target,
            command_buffer,
            image: vk::Image::null(),
            image_ready: vk::Semaphore::null(),
            submit_ready: MaybeOwned::Owned(vk::Semaphore::null()),
            output_size,
        };

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device().begin_command_buffer(ret.command_buffer, &begin_info)
        }.vk("vkBeginCommandBuffer")?;

        let (fmt, extent, framebuffer) = match &ret.target {
            VulkanTarget::Surface(_, swapchain) => {
                // TODO: fixme fuck
                ret.image_ready = unsafe {
                    let info = vk::SemaphoreCreateInfo::builder()
                        .flags(vk::SemaphoreCreateFlags::empty())
                        .build();
                    self.device().create_semaphore(&info, None)
                }.vk("vkCreateSemaphore")?;
                let (image_idx, swap_image) = swapchain.acquire(u64::MAX, From::from(ret.image_ready))
                    .or_else(|e| match e {
                        swapchain::AcquireError::Suboptimal(v) => {
                            warn!(?swapchain, "suboptimal swapchain");
                            Ok(v)
                        },
                        swapchain::AcquireError::Vk(e) => Err(Error::from(e)),
                    })?;
                trace!(image_idx, ?swap_image, "acquired image");
                ret.image = swap_image.image;
                ret.submit_ready = MaybeOwned::Borrowed(swap_image.submit_semaphore);
                let src_layout = if !swap_image.transitioned.fetch_or(true, Ordering::SeqCst) {
                    vk::ImageLayout::UNDEFINED
                } else {
                    vk::ImageLayout::PRESENT_SRC_KHR
                };
                let acquire_flags =
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::COLOR_ATTACHMENT_READ;
                let acquire_barrier = vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(acquire_flags)
                    .old_layout(src_layout)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(swap_image.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    self.device().cmd_pipeline_barrier(
                        ret.command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[], &[], &[acquire_barrier.build()]
                    );
                }
                (swapchain.format(), *swapchain.extent(), swap_image.framebuffer)
            },
        };
        let render_setup = self.render_setups.get(&fmt).unwrap();

        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_setup.render_pass())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&[
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0f32; 4],
                    },
                },
            ]);
        let cb = ret.command_buffer;
        unsafe {
            self.device().cmd_begin_render_pass(cb, &begin_info, vk::SubpassContents::INLINE);
            self.device().cmd_set_viewport(cb, 0, &[
                vk::Viewport {
                    x: 0f32, y: 0f32,
                    width: extent.width as _, height: extent.height as _,
                    min_depth: 0f32, max_depth: 1f32,
                },
            ]);
            self.device().cmd_set_scissor(cb, 0, &[
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }
            ]);
        }
        Ok(ret)
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
    fn bind(
        &mut self,
        target: Rc<crate::backend::vulkan::Surface>,
    ) -> Result<(), <Self as Renderer>::Error> {
        match &self.target {
            Some(tgt) if tgt == &*target => Ok(()),
            _ => {
                use swapchain::SupportDetails;
                let swapchain_support = SupportDetails::with_surface(
                    self.phd.handle(),
                    target.handle(),
                    target.extension(),
                )?;
                let vk::SurfaceFormatKHR { format, color_space } = swapchain_support.choose_format()
                    .ok_or(Error::SwapchainFormat)?;

                self.ensure_render_setup(format)?;
                let render_setup = self.render_setups.get(&format)
                    .ok_or(Error::RendererSetup(format))?;

                self.target = None;
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
        self.target = None;
        Ok(())
    }
}

/// Maximum number of planes supported for imported [`Dmabuf`]s
///
/// Used to statically-size arrays relative to planes to avoid extra allocations being required on
/// each call to [`ImportDma::import_dmabuf`] for [`VulkanRenderer`]
const MAX_PLANES: usize = 4;

impl ImportDma for VulkanRenderer {
    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<<Self as Renderer>::TextureId, <Self as Renderer>::Error> {
        use dmabuf::*;
        const USAGE_FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;
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
            .usage(USAGE_FLAGS)
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
            .usage(USAGE_FLAGS)
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
            dimensions: (width, height),
            format: fmt,
            device: Arc::downgrade(&self.device),
        });
        self.dmabuf_cache.insert(dmabuf.weak(), image.clone());
        Ok(image)
    }

    fn dmabuf_formats(&self) -> Box<dyn Iterator<Item = DrmFormat>> {
        let it = DerefSliceIter::new(self.dmabuf_formats.clone());
        Box::new(it) as Box<_>
    }

    fn has_dmabuf_format(&self, format: DrmFormat) -> bool {
        self.format_for_drm(&format).is_some()
    }
}

impl ImportDmaWl for VulkanRenderer {}

const MEM_FORMATS: [Fourcc; 4] = [
    Fourcc::Argb8888,
    Fourcc::Xrgb8888,
    Fourcc::Abgr8888,
    Fourcc::Xbgr8888,
];

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
        Box::new(MEM_FORMATS.into_iter())
    }
}

impl ImportMemWl for VulkanRenderer {
    fn import_shm_buffer(
        &mut self,
        _buffer: &wayland_server::protocol::wl_buffer::WlBuffer,
        _surface: Option<&crate::wayland::compositor::SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<<Self as Renderer>::TextureId, <Self as Renderer>::Error> {
        todo!()
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

/// Simple wrapper for keeping track of whether or not a given raw vulkan handle is owned or just a
/// borrowed [`Copy`] variant
#[derive(Debug, Clone, Copy)]
enum MaybeOwned<T> {
    Borrowed(T),
    Owned(T),
}

impl<T> Deref for MaybeOwned<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            MaybeOwned::Borrowed(v) => v,
            MaybeOwned::Owned(v) => v,
        }
    }
}

/// [`Frame`] implementation used by a [`VulkanRenderer`]
///
/// * See [`Renderer`] and [`Frame`]
#[derive(Debug)]
pub struct VulkanFrame<'a> {
    renderer: &'a VulkanRenderer,
    target: &'a VulkanTarget,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    image_ready: vk::Semaphore,
    submit_ready: MaybeOwned<vk::Semaphore>,
    output_size: Size<i32, Physical>,
}

// impl<'a> TryFrom<&'a VulkanRenderer> for VulkanFrame<'a> {
//     type Error = <VulkanRenderer as Renderer>::Error;
//     fn try_from(r: &'a VulkanRenderer) -> Result<Self, Self::Error> {
//         use std::mem::MaybeUninit;
//         let target = r.target.as_ref().ok_or(Error::NoTarget)?;
//         let command_buffer = unsafe {
//             let mut ret = MaybeUninit::<vk::CommandBuffer>::zeroed();
//             let alloc_info = vk::CommandBufferAllocateInfo::builder()
//                 .command_pool(*r.command_pool)
//                 .level(vk::CommandBufferLevel::PRIMARY)
//                 .command_buffer_count(1)
//                 .build();
//             let result = (r.device().fp_v1_0().allocate_command_buffers)(
//                 r.device().handle(),
//                 &alloc_info as *const _,
//                 ret.as_mut_ptr(),
//             );
//             match result {
//                 vk::Result::SUCCESS => Ok(ret.assume_init()),
//                 e => Err(Error::Vk {
//                     context: "vkAllocateCommandBuffers",
//                     result: e,
//                 }),
//             }
//         }?;
//         let mut ret = VulkanFrame {
//             renderer: r,
//             target,
//             command_buffer,
//             image: vk::Image::null(),
//             image_ready: vk::Semaphore::null(),
//             submit_ready: MaybeOwned::Owned(vk::Semaphore::null()),
//         };
// 
//         let begin_info = vk::CommandBufferBeginInfo::builder()
//             .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
//         unsafe {
//             r.device().begin_command_buffer(ret.command_buffer, &begin_info)
//         }.vk("vkBeginCommandBuffer")?;
// 
//         let (fmt, extent, framebuffer) = match &ret.target {
//             VulkanTarget::Surface(_, swapchain) => {
//                 // TODO: fixme fuck
//                 ret.image_ready = unsafe {
//                     let info = vk::SemaphoreCreateInfo::builder()
//                         .flags(vk::SemaphoreCreateFlags::empty())
//                         .build();
//                     r.device().create_semaphore(&info, None)
//                 }.vk("vkCreateSemaphore")?;
//                 let (image_idx, swap_image) = swapchain.acquire(u64::MAX, From::from(ret.image_ready))
//                     .or_else(|e| match e {
//                         swapchain::AcquireError::Suboptimal(v) => {
//                             warn!(?swapchain, "suboptimal swapchain");
//                             Ok(v)
//                         },
//                         swapchain::AcquireError::Vk(e) => Err(Error::from(e)),
//                     })?;
//                 trace!(image_idx, ?swap_image, "acquired image");
//                 ret.image = swap_image.image;
//                 ret.submit_ready = MaybeOwned::Borrowed(swap_image.submit_semaphore);
//                 let src_layout = if !swap_image.transitioned.fetch_or(true, Ordering::SeqCst) {
//                     vk::ImageLayout::UNDEFINED
//                 } else {
//                     vk::ImageLayout::PRESENT_SRC_KHR
//                 };
//                 let acquire_flags =
//                     vk::AccessFlags::COLOR_ATTACHMENT_WRITE
//                     | vk::AccessFlags::COLOR_ATTACHMENT_READ;
//                 let acquire_barrier = vk::ImageMemoryBarrier::builder()
//                     .src_access_mask(vk::AccessFlags::empty())
//                     .dst_access_mask(acquire_flags)
//                     .old_layout(src_layout)
//                     .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
//                     .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
//                     .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
//                     .image(swap_image.image)
//                     .subresource_range(vk::ImageSubresourceRange {
//                         aspect_mask: vk::ImageAspectFlags::COLOR,
//                         base_mip_level: 0,
//                         level_count: 1,
//                         base_array_layer: 0,
//                         layer_count: 1,
//                     });
//                 unsafe {
//                     r.device().cmd_pipeline_barrier(
//                         ret.command_buffer,
//                         vk::PipelineStageFlags::TOP_OF_PIPE,
//                         vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
//                         | vk::PipelineStageFlags::FRAGMENT_SHADER,
//                         vk::DependencyFlags::empty(),
//                         &[], &[], &[acquire_barrier.build()]
//                     );
//                 }
//                 (swapchain.format(), *swapchain.extent(), swap_image.framebuffer)
//             },
//         };
//         let render_setup = r.render_setups.get(&fmt).unwrap();
// 
//         let begin_info = vk::RenderPassBeginInfo::builder()
//             .render_pass(render_setup.render_pass())
//             .framebuffer(framebuffer)
//             .render_area(vk::Rect2D {
//                 offset: vk::Offset2D { x: 0, y: 0 },
//                 extent,
//             })
//             .clear_values(&[
//                 vk::ClearValue {
//                     color: vk::ClearColorValue {
//                         float32: [0f32; 4],
//                     },
//                 },
//             ]);
//         let cb = ret.command_buffer;
//         unsafe {
//             r.device().cmd_begin_render_pass(cb, &begin_info, vk::SubpassContents::INLINE);
//             r.device().cmd_set_viewport(cb, 0, &[
//                 vk::Viewport {
//                     x: 0f32, y: 0f32,
//                     width: extent.width as _, height: extent.height as _,
//                     min_depth: 0f32, max_depth: 1f32,
//                 },
//             ]);
//             r.device().cmd_set_scissor(cb, 0, &[
//                 vk::Rect2D {
//                     offset: vk::Offset2D { x: 0, y: 0 },
//                     extent,
//                 }
//             ]);
//         }
//         Ok(ret)
//     }
// }

impl<'a> Drop for VulkanFrame<'a> {
    fn drop(&mut self) {
        let device = self.renderer.device();
        unsafe {
            match &self.submit_ready {
                &MaybeOwned::Owned(v) if v != vk::Semaphore::null() => {
                    device.destroy_semaphore(v, None);
                },
                _ => {},
            }
            if self.command_buffer != vk::CommandBuffer::null() {
                self.renderer.device().free_command_buffers(*self.renderer.command_pool, &[self.command_buffer]);
            }
        }
    }
}

impl<'a> Frame for VulkanFrame<'a> {
    type Error = <VulkanRenderer as Renderer>::Error;
    type TextureId = <VulkanRenderer as Renderer>::TextureId;

    fn id(&self) -> usize {
        self.renderer.id()
    }
    fn clear(&mut self, color: [f32; 4], at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        const DEFAULT_CLEAR_RECT: vk::ClearRect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: 0, height: 0 },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        const fn clear_rect(rect: vk::Rect2D) -> vk::ClearRect {
            vk::ClearRect {
                rect,
                ..DEFAULT_CLEAR_RECT
            }
        }
        let cb = self.command_buffer;
        let device = self.renderer.device();
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: color,
            },
        };
        let clear_attachments = [
            vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: clear_color,
            },
        ];
        let rect_it = at.iter().map(|r| vk::Rect2D {
            offset: vk::Offset2D { x: r.loc.x, y: r.loc.y },
            extent: vk::Extent2D {
                width: r.size.w as _,
                height: r.size.h as _,
            },
        });

        let mut clear_rects_static = [DEFAULT_CLEAR_RECT; 3];
        let clear_rects = if at.len() <= clear_rects_static.len() {
            rect_it.zip(&mut clear_rects_static).for_each(|(rect, out)| {
                out.rect = rect;
            });
            Cow::Borrowed(&clear_rects_static[..at.len()])
        } else {
            let mut rects = Vec::with_capacity(at.len());
            rects.extend(rect_it.map(clear_rect));
            Cow::Owned(rects)
        };
        unsafe {
            device.cmd_clear_attachments(cb, &clear_attachments, &clear_rects);
        }
        Ok(())
    }
    fn draw_solid(
        &mut self,
        _dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _color: [f32; 4],
    ) -> Result<(), Self::Error> {
        // todo!()
        mocked!();
        Ok(())
    }
    fn render_texture_from_to(
        &mut self,
        _texture: &Self::TextureId,
        _src: Rectangle<f64, BufferCoord>,
        _dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _src_transform: Transform,
        _alpha: f32,
    ) -> Result<(), Self::Error> {
        // todo!()
        mocked!();
        Ok(())
    }
    fn transformation(&self) -> Transform {
        Transform::Normal
    }
    fn finish(mut self) -> Result<SyncPoint, Self::Error> {
        let mut submit_fence = VulkanFence::new(self.renderer.device.clone(), false)
            .vk("vkCreateFence")?;

        let cb = self.command_buffer;
        let device = self.renderer.device();
        unsafe {
            device.cmd_end_render_pass(cb);
            device.end_command_buffer(cb)
        }.vk("vkEndCommandBuffer")?;
        unsafe {
            let acquire_stages =
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER;
            let signal_semaphores = [*self.submit_ready];
            let signal_semaphores: &[vk::Semaphore] = if signal_semaphores[0] == vk::Semaphore::null() {
                &[]
            } else {
                &signal_semaphores[..]
            };
            let wait_sems = [self.image_ready];
            let wait_stages = [acquire_stages];
            let cmd_bufs = [self.command_buffer];
            let info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd_bufs)
                .signal_semaphores(signal_semaphores);
            device.queue_submit(
                self.renderer.queues.graphics.handle,
                &[info.build()],
                submit_fence.handle(),
            ).vk("vkQueueSubmit")?;
            if let VulkanTarget::Surface(_, swapchain) = &self.target {
                let wait_sems = signal_semaphores;
                let swapchains = [swapchain.handle()];
                let image_indices = [
                    swapchain.images().iter().position(|swap_img| {
                        swap_img.image == self.image
                    }).map_or(0u32, |idx| idx as _)
                ];
                let mut results = [vk::Result::SUCCESS; 1];
                let info = vk::PresentInfoKHR::builder()
                    .wait_semaphores(wait_sems)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices)
                    .results(&mut results);
                let _suboptimal = swapchain.extension().queue_present(
                    self.renderer.queues.graphics.handle,
                    &info
                ).vk("vkQueuePresentKHR")?;
                results.into_iter()
                    .find_map(|r| std::num::NonZeroI32::new(r.as_raw()))
                    .map_or(Ok(()), |v| Err(vk::Result::from_raw(v.get())))
                    .vk("vkQueuePresentKHR")?;
            }
        }

        // give release-ownership of handles that need to outlive rendering ops to the fence
        //
        // NOTE: if the caller releases the resulting SyncPoint *before* the work is done, that
        // ain't good (for now)
        submit_fence.image_ready = self.image_ready;
        submit_fence.cb = CommandBuffer::new(self.command_buffer, *self.renderer.command_pool);
        self.image_ready = vk::Semaphore::null();
        self.command_buffer = vk::CommandBuffer::null();

        Ok(SyncPoint::from(submit_fence))
    }
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.renderer.device.wait_fence_vk(sync)
    }
}

#[derive(Debug)]
struct Format {
    vk: vk::Format,
    drm: Fourcc,
    modifier: Option<vk::DrmFormatModifierPropertiesEXT>,
}

/// [`Texture`] implementation for a [`VulkanRenderer`]
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct VulkanImage(Arc<InnerImage>);

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
    dimensions: (u32, u32),
    format: Format,
    device: Weak<Device>,
}

impl InnerImage {
    #[inline(always)]
    fn memories(&self) -> impl Iterator<Item=&'_ vk::DeviceMemory> {
        self.memories.iter()
            .filter(|&&mem| mem != vk::DeviceMemory::null())
    }
}

impl Drop for InnerImage {
    fn drop(&mut self) {
        let Some(device) = self.device.upgrade() else {
            error!("device destroyed before image: {:?}", self.image);
            return;
        };
        unsafe {
            device.destroy_image(self.image, None);
            for &mem in self.memories() {
                device.free_memory(mem, None);
            }
        }
    }
}

struct Extensions {
    image_format_modifier: ImageDrmFormatModifier,
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
            image_format_modifier: ImageDrmFormatModifier::new(i, d),
            external_memory_fd: ExternalMemoryFd::new(i, d),
        }
    }
}

#[derive(Debug)]
struct FormatInfo {
    vk: vk::Format,
    drm: Fourcc,
    properties: vk::FormatProperties,
    modifiers: Arc<[vk::DrmFormatModifierPropertiesEXT]>,
}

impl FormatInfo {
    fn get_known(phd: &PhysicalDevice) -> impl Iterator<Item = Self> + '_ {
        use crate::backend::allocator::vulkan::format;
        format::known_vk_formats().map(|(fourcc, vk_format)| Self::new(phd, fourcc, vk_format))
    }
    fn new(phd: &PhysicalDevice, fourcc: Fourcc, vk_format: vk::Format) -> Self {
        let instance = phd.instance().handle();
        let mut mod_list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut props = vk::FormatProperties2::builder().push_next(&mut mod_list).build();
        unsafe {
            instance.get_physical_device_format_properties2(phd.handle(), vk_format, &mut props);
        }
        let mut mod_props = Vec::with_capacity(mod_list.drm_format_modifier_count as _);
        mod_list.p_drm_format_modifier_properties = mod_props.as_mut_ptr();
        unsafe {
            instance.get_physical_device_format_properties2(phd.handle(), vk_format, &mut props);
            mod_props.set_len(mod_list.drm_format_modifier_count as _);
        }
        FormatInfo {
            vk: vk_format,
            drm: fourcc,
            properties: props.format_properties,
            modifiers: mod_props.into(),
        }
    }
}


#[derive(Debug)]
struct QueueFamilies {
    properties: Vec<vk::QueueFamilyProperties>,
    graphics: Queue,
}

impl<'a> TryFrom<&'a PhysicalDevice> for QueueFamilies {
    type Error = Error<'static>;
    fn try_from(pd: &'a PhysicalDevice) -> Result<Self, Self::Error> {
        let props = unsafe {
            pd.instance()
                .handle()
                .get_physical_device_queue_family_properties(pd.handle())
        };
        let gfx = props.iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(Queue::with_index)
            .ok_or(Error::NoGraphicsQueue)?;
        Ok(QueueFamilies {
            properties: props,
            graphics: gfx,
        })
    }
}

impl QueueFamilies {
    fn fill_device(&mut self, device: &ash::Device) {
        self.graphics.fill_handle(device, 0);
    }
    fn present_queue(
        &self,
        pd: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        ext: &ash::extensions::khr::Surface
    ) -> Result<usize, Error<'static>> {
        if let Some(&Queue { index, .. }) = [&self.graphics].into_iter()
            .find(|&q| q.can_present(surface, ext, pd).ok() == Some(true)) {
            return Ok(index);
        }
        let it = [
            &self.graphics
        ].into_iter()
            .map(|q| q.can_present(surface, ext, pd).map(|v| (q.index, v)))
            .chain({
                (0..self.properties.len())
                    .map(|idx| {
                        Queue::with_index(idx)
                            .can_present(surface, ext, pd)
                            .map(|v| (idx, v))
                    })
            });
        for r in it {
            let (idx, can_present) = r?;
            if can_present {
                return Ok(idx);
            }
        }
        Err(Error::NoPresentQueue)
    }
}

#[derive(Debug, Clone, Copy)]
struct Queue {
    index: usize,
    handle: vk::Queue,
}

impl Queue {
    #[inline(always)]
    const fn with_index(index: usize) -> Self {
        Queue {
            index,
            handle: vk::Queue::null(),
        }
    }

    fn fill_handle(&mut self, device: &ash::Device, queue_index: u32) {
        let info = vk::DeviceQueueInfo2::builder()
            .flags(vk::DeviceQueueCreateFlags::empty())
            .queue_family_index(self.index as _)
            .queue_index(queue_index);
        self.handle = unsafe { device.get_device_queue2(&info) };
    }

    #[inline]
    fn can_present(&self, surface: vk::SurfaceKHR, ext: &ash::extensions::khr::Surface, pd: vk::PhysicalDevice) -> Result<bool, Error<'static>> {
        unsafe {
            ext.get_physical_device_surface_support(pd, self.index as _, surface)
        }.vk("vkGetPhysicalDeviceSurfaceSupportKHR")
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
    /// Occurs when [`Renderer::render`] is called without a target bound for the renderer
    #[error("no render target bound")]
    NoTarget,
    /// Occurs if a new rendering frame is started, but all of the available command buffers are
    /// already busy
    #[error("all command buffers already busy")]
    AllCommandBuffersBusy,
    #[error("swapchain support not found for surface")]
    SwapchainSupport,
    #[error("could not find compatible swapchain format")]
    SwapchainFormat,
    #[error("unimplemented: {0}")]
    Unimplemented(&'a str),
    #[error("no renderer setup found for format: {0:?}")]
    RendererSetup(vk::Format),
    #[error("waiting for fence interrupted: {0}")]
    Interrupted(#[from] crate::backend::renderer::sync::Interrupted),
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

trait BufferExtVulkan {
    fn extent_2d(&self) -> vk::Extent2D;
    #[inline(always)]
    fn extent_3d(&self) -> vk::Extent3D {
        vk::Extent3D::from(self.extent_2d())
    }
}
impl<T: Buffer> BufferExtVulkan for T {
    #[inline(always)]
    fn extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width(),
            height: self.height(),
        }
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
}

contextual_handles!(Device {
    vk::CommandPool = destroy_command_pool,
    vk::RenderPass = destroy_render_pass,
});

vulkan_handles! {
    vk::CommandBuffer,
    vk::DeviceMemory,
}
