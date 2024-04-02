use ash::{
    vk,
    extensions::{
        ext::ImageDrmFormatModifier,
        khr::ExternalMemoryFd,
    },
};
use crate::{
    backend::{
        allocator::{dmabuf::{Dmabuf, WeakDmabuf}, Buffer, Format as DrmFormat, Fourcc, Modifier as DrmModifier},
        drm::DrmNode,
        renderer::{sync::SyncPoint, DebugFlags, Frame, Renderer, Texture, TextureFilter},
        vulkan::{
            version::Version,
            Instance,
            PhysicalDevice,
        },
    },
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
};
use std::{
    collections::HashMap,
    ffi::CStr,
    fmt,
    os::fd::{
        BorrowedFd,
        AsRawFd,
        IntoRawFd,
    },
    ops::{
        Deref,
        DerefMut,
    },
    rc::Rc,
    sync::{
        Arc,
        Weak,
    },
};
use super::ImportDma;
use tracing::{trace, error};
use scopeguard::ScopeGuard;

mod dmabuf;
mod util;
use util::*;

#[derive(Debug)]
pub struct VulkanRenderer {
    phd: PhysicalDevice,
    device: Arc<Device>,
    queue: (usize, vk::Queue),
    node: Option<DrmNode>,
    formats: HashMap<Fourcc, FormatInfo>,
    dmabuf_formats: Rc<[DrmFormat]>,
    extensions: Extensions,
    debug_flags: DebugFlags,
    dmabuf_cache: HashMap<WeakDmabuf, VulkanImage>,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    target: Option<VulkanTarget>,
}

impl VulkanRenderer {
    fn required_extensions(phd: &PhysicalDevice) -> &'static [*const i8] {
        const EXTS: &[*const i8] = &[
            // Always
            vk::ExtImageDrmFormatModifierFn::name().as_ptr(),
            vk::ExtExternalMemoryDmaBufFn::name().as_ptr(),
            vk::KhrExternalMemoryFdFn::name().as_ptr(),
            // < 1.2
            vk::KhrImageFormatListFn::name().as_ptr(),
        ];
        if phd.api_version() >= Version::VERSION_1_2 {
            &EXTS[..3]
        } else {
            EXTS
        }
    }

    /// create a new [`VulkanRenderer`] from this [`PhysicalDevice`]
    pub fn new(phd: &PhysicalDevice) -> Result<Self, Error<'static>> {
        let queue_family_idx = phd.graphics_queue_family()
            .ok_or(Error::NoGraphicsQueue)?;
        let priorities = [0.0];
        let queue_info = [
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_idx as u32)
                .queue_priorities(&priorities)
                .build(),
        ];
        let create_info = vk::DeviceCreateInfo::builder()
            .enabled_extension_names(Self::required_extensions(phd))
            .queue_create_infos(&queue_info);
        let node = phd
            .render_node().ok().flatten()
            .or_else(|| phd.primary_node().ok().flatten());
        let instance = phd.instance().handle();
        let device = unsafe {
            instance.create_device(phd.handle(), &create_info, None)
        }.map_or_else(
            |e| Err(Error::Vk {
                context: "vkCreateDevice",
                result: e,
            }),
            |d| Ok(Device(d)),
        )?;

        let queue_info = vk::DeviceQueueInfo2::builder()
            .flags(vk::DeviceQueueCreateFlags::empty())
            .queue_family_index(queue_family_idx as _)
            .queue_index(0);
        let queue = unsafe {
            device.get_device_queue2(&queue_info)
        };

        let extensions = Extensions::new(instance, &device);

        let formats: HashMap<_, _> = FormatInfo::get_known(phd)
            .map(|f| (f.drm, f))
            .collect();
        let dmabuf_formats = formats.iter()
            .flat_map(|(_, &FormatInfo { drm, ref modifiers, .. })| {
                modifiers.iter()
                    .map(move |v| DrmFormat {
                        code: drm,
                        modifier: DrmModifier::from(v.drm_format_modifier),
                    })
            })
            .collect::<Vec<_>>();

        let memory_props = unsafe {
            instance.get_physical_device_memory_properties(phd.handle())
        };

        Ok(VulkanRenderer {
            phd: phd.clone(),
            device: Arc::new(device),
            queue: (queue_family_idx, queue),
            node,
            formats,
            dmabuf_formats: dmabuf_formats.into(),
            extensions,
            debug_flags: DebugFlags::empty(),
            dmabuf_cache: HashMap::new(),
            memory_props,
            target: None,
        })
    }

    #[inline(always)]
    fn instance(&self) -> &ash::Instance {
        self.phd.instance().handle()
    }

    fn format_for_drm(&self, format: &DrmFormat) -> Option<Format> {
        let &DrmFormat { code , modifier } = format;
        self.formats.get(&code).and_then(|info| {
            info.modifiers.iter()
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
        self.memory_props.memory_types.iter()
            .enumerate()
            .filter(|&(idx, ..)| (type_mask & (0b1 << idx)) != 0)
            .find(|(_idx, ty)| (ty.property_flags & props) == props)
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
    fn downscale_filter(
        &mut self,
        _filter: TextureFilter
    ) -> Result<(), Self::Error> {
        todo!()
    }
    fn upscale_filter(
        &mut self,
        _filter: TextureFilter
    ) -> Result<(), Self::Error> {
        todo!()
    }
    fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.debug_flags = flags;
    }
    fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }
    fn render(
        &mut self,
        _output_size: Size<i32, Physical>,
        _dst_transform: Transform
    ) -> Result<Self::Frame<'_>, Self::Error> {
        let _target = self.target.as_ref()
            .ok_or(Error::NoTarget)?;
        todo!()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum VulkanTarget {
    Framebuffer(vk::Framebuffer),
}

const MAX_PLANES: usize = 4;

impl ImportDma for VulkanRenderer {
    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>
    ) -> Result<<Self as Renderer>::TextureId, <Self as Renderer>::Error> {
        use dmabuf::*;
        const USAGE_FLAGS: vk::ImageUsageFlags =
            vk::ImageUsageFlags::SAMPLED;
        const H_TYPE: vk::ExternalMemoryHandleTypeFlags =
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        const ALL_PLANE_ASPECTS: [vk::ImageAspectFlags; MAX_PLANES] = [
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
        ];

        if let Some(existing) = self.dmabuf_cache
            .get_key_value(&dmabuf.weak())
            .and_then(|(weak, img)| {
                weak.upgrade()
                    .filter(|buf| buf == dmabuf)?;
                Some(img.clone())
            }) {
            return Ok(existing);
        }

        let fmt = self.format_for_drm(&dmabuf.format())
            .ok_or(Error::UnknownFormat(dmabuf.format()))?;

        let disjoint = dmabuf.is_disjoint()?;
        let image_create_flags = if disjoint {
            vk::ImageCreateFlags::DISJOINT
        } else {
            vk::ImageCreateFlags::empty()
        };
        let mut external_info = vk::PhysicalDeviceExternalImageFormatInfo::builder()
            .handle_type(H_TYPE);
        let mut info = vk::PhysicalDeviceImageFormatInfo2::builder()
            .format(fmt.vk)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(USAGE_FLAGS)
            .flags(image_create_flags)
            .push_next(&mut external_info);
        let mut drm_info: vk::PhysicalDeviceImageDrmFormatModifierInfoEXT;
        let queue_indices = [self.queue.0 as u32];
        if let Some(mod_info) = fmt.modifier.as_ref() {
            drm_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::builder()
                .drm_format_modifier(mod_info.drm_format_modifier)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&queue_indices)
                .build();
            info = info.push_next(&mut drm_info);
        }
        let mut external_fmt_props = vk::ExternalImageFormatProperties::default();
        let mut image_fmt = vk::ImageFormatProperties2::builder()
            .push_next(&mut external_fmt_props);
        let _image_fmt = unsafe {
            self.instance().get_physical_device_image_format_properties2(
                self.phd.handle(),
                &info,
                &mut image_fmt
            ).map_err(Error::vk("vkGetPhysicalDeviceImageFormatProperties2"))?;
            image_fmt.image_format_properties
        };

        let mut drm_info: vk::ImageDrmFormatModifierExplicitCreateInfoEXTBuilder<'_>;
        let mut plane_layouts: Vec<vk::SubresourceLayout>;
        let mut external_info = vk::ExternalMemoryImageCreateInfo::builder()
            .handle_types(H_TYPE);
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
            let it = dmabuf.offsets()
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
        let image = unsafe {
            self.device.create_image(&info, None)
        }.map_or_else(
            |e| Err(Error::Vk {
                context: "vkCreateImage",
                result: e,
            }),
            |img| Ok(scopeguard::guard(img, |img| unsafe {
                self.device.destroy_image(img, None);
            }))
        )?;
        trace!("imported VkImage: {:?}", &image);

        let memory_plane_props = self.extensions.external_memory_fd
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
                if m != vk::DeviceMemory::null() {
                    self.device.free_memory(m, None);
                }
            }
        });
        memory_plane_props.into_iter()
            .enumerate()
            .zip(memories.iter_mut())
            .take(mem_count)
            .try_for_each(|((idx, (fd, props)), dm)| -> Result<_, Error<'static>> {
                let mem_reqs = self.device.get_memory_requirements(
                    *image,
                    ALL_PLANE_ASPECTS[idx],
                    disjoint
                ).memory_requirements;
                let mem_type_bits =
                    props.memory_type_bits & mem_reqs.memory_type_bits;
                let (mem_idx, ..) = self.find_memory_type(
                    mem_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                ).ok_or(DmabufError::NoMemoryType(mem_type_bits))?;
                let fd = fd.try_clone_to_owned()
                    .map_err(|e| Error::from(DmabufError::Io(e)))?;
                let mut import_info = vk::ImportMemoryFdInfoKHR::builder()
                    .handle_type(H_TYPE)
                    .fd(fd.as_raw_fd());
                let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::builder()
                    .image(*image);
                let create_info = vk::MemoryAllocateInfo::builder()
                    .allocation_size(mem_reqs.size)
                    .memory_type_index(mem_idx as _)
                    .push_next(&mut dedicated_info)
                    .push_next(&mut import_info);
                *dm = unsafe {
                    self.device.allocate_memory(&create_info, None)
                }.map_err(Error::vk("vkAllocateMemory"))?;
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
        trace!("import_dmabuf: imported device memories[{}]:\n{:#?}", mems.len(), mems);

        unsafe {
            self.device.bind_image_memory2(&bind_infos[..mems.len()])
        }.map_err(Error::vk("vkBindImageMemory2"))?;

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

pub struct VulkanFrame<'a> {
    renderer: &'a VulkanRenderer,
}

impl<'a> Frame for VulkanFrame<'a> {
    type Error = <VulkanRenderer as Renderer>::Error;
    type TextureId = <VulkanRenderer as Renderer>::TextureId;

    fn id(&self) -> usize {
        self.renderer.id()
    }
    fn clear(
        &mut self,
        color: [f32; 4],
        at: &[Rectangle<i32, Physical>]
    ) -> Result<(), Self::Error> {
        todo!()
    }
    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: [f32; 4]
    ) -> Result<(), Self::Error> {
        todo!()
    }
    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32
    ) -> Result<(), Self::Error> {
        todo!()
    }
    fn transformation(&self) -> Transform {
        Transform::Normal
    }
    fn finish(self) -> Result<SyncPoint, Self::Error> {
        todo!()
    }
}

#[derive(Debug)]
struct Format {
    vk: vk::Format,
    drm: Fourcc,
    modifier: Option<vk::DrmFormatModifierPropertiesEXT>,
}

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

impl Drop for InnerImage {
    fn drop(&mut self) {
        let Some(device) = self.device.upgrade() else {
            error!("device destroyed before image: {:?}", self.image);
            return;
        };
        unsafe {
            device.destroy_image(self.image, None);
            for &mem in self.memories.iter().filter(|&&mem| mem != vk::DeviceMemory::null()) {
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

fn known_formats() -> impl Iterator<Item=(Fourcc, vk::Format)> {
    use crate::backend::allocator::vulkan::format;
    format::known_formats()
        .iter()
        .copied()
        .filter_map(|f| format::get_vk_format(f).map(move |vk| (f, vk)))
}

#[derive(Debug)]
struct FormatInfo {
    vk: vk::Format,
    drm: Fourcc,
    properties: vk::FormatProperties,
    modifiers: Arc<[vk::DrmFormatModifierPropertiesEXT]>,
}

impl FormatInfo {
    fn get_known(phd: &PhysicalDevice) -> impl Iterator<Item=Self> + '_ {
        known_formats().map(|(fourcc, vk_format)| {
            Self::new(phd, fourcc, vk_format)
        })
    }
    fn new(phd: &PhysicalDevice, fourcc: Fourcc, vk_format: vk::Format) -> Self {
        let instance = phd.instance().handle();
        let mut mod_list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut props = vk::FormatProperties2::builder()
            .push_next(&mut mod_list)
            .build();
        unsafe {
            instance.get_physical_device_format_properties2(
                phd.handle(),
                vk_format,
                &mut props
            );
        }
        let mut mod_props = Vec::with_capacity(mod_list.drm_format_modifier_count as _);
        mod_list.p_drm_format_modifier_properties = mod_props.as_mut_ptr();
        unsafe {
            instance.get_physical_device_format_properties2(
                phd.handle(),
                vk_format,
                &mut props
            );
            mod_props.set_len(mod_list.p_drm_format_modifier_properties as _);
        }
        FormatInfo {
            vk: vk_format,
            drm: fourcc,
            properties: props.format_properties,
            modifiers: mod_props.into(),
        }
    }
}

trait PdExt {
    fn graphics_queue_family(&self) -> Option<usize>;
}
impl PdExt for PhysicalDevice {
    fn graphics_queue_family(&self) -> Option<usize> {
        let queue_families = unsafe {
            self.instance()
                .handle()
                .get_physical_device_queue_family_properties(self.handle())
        };
        queue_families.iter()
            .position(|props| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
    }
}

#[repr(transparent)]
struct Device(ash::Device);
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
        f.debug_tuple("Device")
            .field(&self.0.handle())
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error<'a> {
    #[error("physical device had no graphics queue family")]
    NoGraphicsQueue,
    #[error("{context}: {result}")]
    Vk {
        context: &'a str,
        result: vk::Result,
    },
    #[error("error importing dmabuf: {0}")]
    Dmabuf(#[from] dmabuf::DmabufError),
    #[error("failed to convert import format: {0:?}")]
    UnknownFormat(DrmFormat),
    #[error("no render target bound")]
    NoTarget,
}

impl<'a> Error<'a> {
    #[inline(always)]
    fn vk(s: &'a str) -> impl Fn(vk::Result) -> Self + 'a {
        move |e: vk::Result| Error::Vk {
            context: s,
            result: e,
        }
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
        plane_aspect: vk::ImageAspectFlags
    ) -> vk::MemoryRequirements2;

    #[inline(always)]
    fn get_memory_requirements(
        &self,
        image: vk::Image,
        plane_aspect: vk::ImageAspectFlags,
        disjoint: bool
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
        let info = vk::ImageMemoryRequirementsInfo2::builder()
            .image(image);
        let mut ret = vk::MemoryRequirements2::default();
        unsafe {
            self.get_image_memory_requirements2(&info, &mut ret);
        }
        ret
    }
    fn disjoint_memory_requirements(
        &self,
        image: vk::Image,
        plane_aspect: vk::ImageAspectFlags
    ) -> vk::MemoryRequirements2 {
        let mut info_plane = vk::ImagePlaneMemoryRequirementsInfo::builder()
            .plane_aspect(plane_aspect);
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
