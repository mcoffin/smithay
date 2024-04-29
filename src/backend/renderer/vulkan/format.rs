use ash::vk;
use crate::backend::{
    allocator::Fourcc,
    vulkan::PhysicalDevice,
};
use std::sync::Arc;
use super::Error;
use wayland_server::protocol::wl_shm;

#[derive(Debug)]
pub struct FormatInfo {
    pub vk: vk::Format,
    pub drm: Fourcc,
    pub properties: vk::FormatProperties,
    pub modifiers: Arc<[vk::DrmFormatModifierPropertiesEXT]>,
}

impl FormatInfo {
    pub fn get_known(phd: &PhysicalDevice) -> impl Iterator<Item = Self> + '_ {
        use crate::backend::allocator::vulkan::format;
        format::known_vk_formats().map(|(fourcc, vk_format)| Self::new(phd, fourcc, vk_format))
    }
    pub fn new(phd: &PhysicalDevice, fourcc: Fourcc, vk_format: vk::Format) -> Self {
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
pub struct Format {
    pub vk: vk::Format,
    pub drm: Fourcc,
    pub modifier: Option<vk::DrmFormatModifierPropertiesEXT>,
}

impl TryFrom<wl_shm::Format> for Format {
    type Error = Error<'static>;
    fn try_from(shm_format: wl_shm::Format) -> Result<Self, Self::Error> {
        use crate::backend::allocator::vulkan::format;
        use crate::wayland::shm::shm_format_to_fourcc;
        let format = shm_format_to_fourcc(shm_format)
            .ok_or(Error::UnknownShmFormat(shm_format))?;
        let vk_format = format::known_vk_formats()
            .find_map(|(fmt, vk_fmt)| if format == fmt {
                Some(vk_fmt)
            } else {
                None
            })
            .ok_or(Error::UnknownMemoryFormat(format))?;
        Ok(Format {
            vk: vk_format,
            drm: format,
            modifier: None,
        })
    }
}
