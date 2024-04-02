#![allow(missing_docs)]
//! Abstractions that hopefully get rid of many of the platform-specific interactions w/ the vulkan
//! WSI.
//!
//! EX: `vkCreateWin32SurfaceKHR` vs. `vkCreateWaylandSurfaceKHR`
use ash::{extensions::khr::WaylandSurface, vk};
use std::{ffi::CStr, marker::PhantomData, os::raw::c_void, ptr::NonNull};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub trait VulkanNativeWindow<'a> {
    fn required_extensions(&self) -> &'a [&'a CStr];
    fn create_surface(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<vk::SurfaceKHR, vk::Result>;
}

#[derive(Debug, Clone, Copy)]
struct VulkanNativeWayland<'a> {
    display: NonNull<vk::wl_display>,
    surface: NonNull<vk::wl_surface>,
    _marker: PhantomData<&'a c_void>,
}

impl<'a> VulkanNativeWindow<'static> for VulkanNativeWayland<'a> {
    #[inline(always)]
    fn required_extensions(&self) -> &'static [&'static CStr] {
        const EXTS: &[&CStr] = &[vk::KhrSurfaceFn::name(), vk::KhrWaylandSurfaceFn::name()];
        EXTS
    }

    fn create_surface(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<vk::SurfaceKHR, vk::Result> {
        let ext = WaylandSurface::new(entry, instance);
        let info = vk::WaylandSurfaceCreateInfoKHR::builder()
            .display(self.display.as_ptr())
            .surface(self.surface.as_ptr());
        unsafe { ext.create_wayland_surface(&info, None) }
    }
}

pub trait TryVulkanNativeWindow {
    fn vulkan_native_window(&self) -> Option<Box<dyn VulkanNativeWindow<'static> + '_>>;
}

impl<W> TryVulkanNativeWindow for W
where
    W: HasDisplayHandle + HasWindowHandle,
{
    fn vulkan_native_window(&self) -> Option<Box<dyn VulkanNativeWindow<'static> + '_>> {
        use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};
        let (d, w) = self
            .display_handle()
            .and_then(|d| self.window_handle().map(move |w| (d, w)))
            .ok()?;
        match (d.as_raw(), w.as_raw()) {
            (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
                Some(Box::new(VulkanNativeWayland {
                    display: display.display.cast(),
                    surface: window.surface.cast(),
                    _marker: PhantomData,
                }) as Box<_>)
            }
            _ => None,
        }
    }
}
