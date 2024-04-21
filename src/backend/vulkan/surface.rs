use super::Instance;
use ash::{extensions::khr::Surface as VkSurface, vk};
use std::{fmt, ops::Deref};

pub struct Surface {
    handle: vk::SurfaceKHR,
    surface: VkSurface,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface.destroy_surface(self.handle, None);
        }
    }
}

impl Surface {
    #[inline(always)]
    pub fn new(ext: VkSurface, handle: vk::SurfaceKHR) -> Self {
        Surface { handle, surface: ext }
    }
    #[inline(always)]
    pub fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    #[inline(always)]
    pub fn extension(&self) -> &VkSurface {
        &self.surface
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
