use ash::{extensions::khr::Surface as VkSurface, vk};
use std::{fmt, ops::Deref, cell::Cell};

/// *Owned* version of [`vk::SurfaceKHR`] that will be destroyed when dropped
pub struct Surface {
    handle: vk::SurfaceKHR,
    surface: VkSurface,
    extent: Cell<vk::Extent2D>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface.destroy_surface(self.handle, None);
        }
    }
}

impl Surface {
    /// Creates a new vulkan surface given a loaded extension and a [`vk::SurfaceKHR`] handle
    #[inline(always)]
    pub fn new(ext: VkSurface, handle: vk::SurfaceKHR) -> Self {
        Surface {
            handle,
            surface: ext,
            extent: Cell::new(vk::Extent2D {
                width: 0,
                height: 0,
            }),
        }
    }

    /// Gives access to the raw vulkan handle value
    #[inline(always)]
    pub fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    /// Gets a reference to the extension vtable
    #[inline(always)]
    pub fn extension(&self) -> &VkSurface {
        &self.surface
    }

    /// Gives the size (width, height) of the surface (in pixels)
    #[inline(always)]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent.get()
    }

    #[inline(always)]
    pub(crate) fn resize(&self, width: u32, height: u32) {
        self.extent.set(vk::Extent2D { width, height });
    }
}

impl Deref for Surface {
    type Target = vk::SurfaceKHR;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl fmt::Debug for Surface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Surface").field(&self.handle).finish()
    }
}
