use ash::vk;
use crate::backend::vulkan::PhysicalDevice;
use super::{
    Error,
    ErrorExt,
};

#[derive(Debug)]
pub struct QueueFamilies {
    properties: Vec<vk::QueueFamilyProperties>,
    pub graphics: Queue,
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
    pub fn fill_device(&mut self, device: &ash::Device) {
        self.graphics.fill_handle(device, 0);
    }
    pub fn present_queue(
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

    #[allow(dead_code)]
    #[inline(always)]
    pub fn properties(&self) -> &[vk::QueueFamilyProperties] {
        self.properties.as_slice()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Queue {
    pub index: usize,
    pub handle: vk::Queue,
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

    #[inline(always)]
    pub fn idx(&self) -> u32 {
        self.index as _
    }
}
