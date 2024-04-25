use super::{ImportDma, ImportDmaWl, ImportEgl, ImportMem, ImportMemWl};
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

mod dmabuf;
mod swapchain;
mod util;
use util::*;
use swapchain::Swapchain;

const DEFAULT_COMMAND_BUFFERS: usize = 4;

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
    command_buffers: Box<[vk::CommandBuffer]>,
    command_buffer_usage: Box<[AtomicBool]>,
    target: Option<VulkanTarget>,
    upscale_filter: vk::Filter,
    downscale_filter: vk::Filter,
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
        }
        .vk("vkCreateCommandPool")?;

        let cmd_buf_alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(DEFAULT_COMMAND_BUFFERS as _);
        let cmd_buffers =
            unsafe { device.allocate_command_buffers(&cmd_buf_alloc_info) }.vk("vkAllocateCommandBuffers")?;

        let cmd_buffer_usage = {
            let mut ret = Vec::with_capacity(cmd_buffers.len());
            let usage_it = std::iter::repeat_with(|| AtomicBool::new(false));
            ret.extend(usage_it.take(cmd_buffers.len()));
            ret.into()
        };

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
            command_buffers: cmd_buffers.into(),
            command_buffer_usage: cmd_buffer_usage,
            target: None,
            upscale_filter: vk::Filter::LINEAR,
            downscale_filter: vk::Filter::LINEAR,
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

    #[inline(always)]
    fn command_buffers(&self) -> impl Iterator<Item = (vk::CommandBuffer, &AtomicBool)> {
        self.command_buffers
            .iter()
            .copied()
            .zip(&*self.command_buffer_usage)
    }

    /// TODO: verify this AtomicBool logic and ordering w/ [`CommandBuffer::drop`]
    fn acquire_command_buffer(&self) -> Result<CommandBuffer<'_>, Error<'static>> {
        self.command_buffers()
            .find(|&(_buf, in_use)| {
                in_use
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::Acquire)
                    .is_ok()
            })
            .map(CommandBuffer::from)
            .ok_or(Error::AllCommandBuffersBusy)
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        // TODO: consider moving this into some kind of wrapper?
        unsafe {
            if self.command_buffers.len() > 0 && self.command_buffers.iter().any(|&buf| !buf.is_null()) {
                self.device
                    .free_command_buffers(*self.command_pool, &self.command_buffers);
            }
        }
    }
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
        _output_size: Size<i32, Physical>,
        _dst_transform: Transform,
    ) -> Result<Self::Frame<'_>, Self::Error> {
        Ok(VulkanFrame {
            renderer: self,
            target: self.target.as_ref().ok_or(Error::NoTarget)?,
            command_buffer: self.acquire_command_buffer()?,
        })
    }
    fn wait(&mut self, _sync: &SyncPoint) -> Result<(), Self::Error> {
        todo!()
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
                let surface = target.handle();
                let swapchain = Swapchain::with_surface(
                    &*self,
                    target.handle(),
                    target.extension(),
                    &target.extent(),
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                )?;
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

impl ImportEgl for VulkanRenderer {
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

/// [`Frame`] implementation used by a [`VulkanRenderer`]
///
/// * See [`Renderer`] and [`Frame`]
#[derive(Debug)]
pub struct VulkanFrame<'a> {
    renderer: &'a VulkanRenderer,
    target: &'a VulkanTarget,
    command_buffer: CommandBuffer<'a>,
}

impl<'a> Drop for VulkanFrame<'a> {
    fn drop(&mut self) {
        trace!("{}", fn_name!());
        todo!();
    }
}

impl<'a> Frame for VulkanFrame<'a> {
    type Error = <VulkanRenderer as Renderer>::Error;
    type TextureId = <VulkanRenderer as Renderer>::TextureId;

    fn id(&self) -> usize {
        self.renderer.id()
    }
    fn clear(&mut self, _color: [f32; 4], _at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        todo!()
    }
    fn draw_solid(
        &mut self,
        _dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _color: [f32; 4],
    ) -> Result<(), Self::Error> {
        todo!()
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
        todo!()
    }
    fn transformation(&self) -> Transform {
        Transform::Normal
    }
    fn finish(self) -> Result<SyncPoint, Self::Error> {
        todo!()
    }
    fn wait(&mut self, _sync: &SyncPoint) -> Result<(), Self::Error> {
        todo!()
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
    unsafe fn allocate_command_buffers_array(
        &self,
        info: &mut vk::CommandBufferAllocateInfo,
        buffers: &mut [vk::CommandBuffer],
    ) -> Result<(), vk::Result>;
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
    unsafe fn allocate_command_buffers_array(
        &self,
        info: &mut vk::CommandBufferAllocateInfo,
        buffers: &mut [vk::CommandBuffer],
    ) -> Result<(), vk::Result> {
        let create = self.fp_v1_0().allocate_command_buffers;
        info.command_buffer_count = buffers.len() as _;
        let ret = create(self.handle(), info, buffers.as_mut_ptr());
        if ret == vk::Result::SUCCESS {
            Ok(())
        } else {
            Err(ret)
        }
    }
}

/// Borrowed [`vk::CommandBuffer`] from a [`VulkanRenderer`]
///
/// Will set it's `in_use` flag back to `false` in the [`Drop`] implementation
#[derive(Debug)]
struct CommandBuffer<'a> {
    handle: vk::CommandBuffer,
    in_use: &'a AtomicBool,
}

impl<'a> Drop for CommandBuffer<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        trace!("{}", fn_name!());
        // in release mode, just store the value as `false`, assuming that the previous value was
        // `true`
        #[cfg(not(debug_assertions))]
        self.in_use.store(false, Ordering::Relaxed);

        // in debug mode, use `compare_exchange` to ensure the value was previously `true`
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.in_use
                    .compare_exchange(true, false, Ordering::SeqCst, Ordering::Acquire),
                Ok(true)
            )
        }
    }
}

impl<'a> From<(vk::CommandBuffer, &'a AtomicBool)> for CommandBuffer<'a> {
    #[inline(always)]
    fn from(v: (vk::CommandBuffer, &'a AtomicBool)) -> Self {
        let (handle, in_use) = v;
        CommandBuffer { handle, in_use }
    }
}

contextual_handles!(Device {
    vk::CommandPool = destroy_command_pool,
});

vulkan_handles! {
    vk::CommandBuffer,
    vk::DeviceMemory,
}
