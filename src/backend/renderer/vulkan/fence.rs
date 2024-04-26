use ash::vk;
use std::sync::Arc;
use crate::{
    fn_name,
    backend::renderer::sync::Fence,
};
use tracing::*;

#[derive(Debug)]
pub struct VulkanFence {
    device: Arc<super::Device>,
    handle: vk::Fence,
}

impl VulkanFence {
    pub fn new(device: Arc<super::Device>, signaled: bool) -> Result<Self, vk::Result> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let info = vk::FenceCreateInfo::builder()
            .flags(flags)
            .build();
        let handle = unsafe {
            device.create_fence(&info, None)
        }?;
        Ok(VulkanFence {
            device,
            handle,
        })
    }

    #[inline(always)]
    pub fn handle(&self) -> vk::Fence {
        self.handle
    }
}

impl Drop for VulkanFence {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.handle, None);
        }
    }
}

impl Fence for VulkanFence {
    fn is_signaled(&self) -> bool {
        unsafe {
            self.device.get_fence_status(self.handle)
        }.unwrap_or_else(|e| {
            error!("{}: {:?}", fn_name!(), e);
            true
        })
    }

    fn wait(&self) -> Result<(), crate::backend::renderer::sync::Interrupted> {
        use crate::backend::renderer::sync::Interrupted;
        unsafe {
            self.device.wait_for_fences(&[self.handle], true, u64::MAX)
        }.map_err(|_| Interrupted)
    }

    fn is_exportable(&self) -> bool {
        // todo!()
        false
    }

    fn export(&self) -> Option<std::os::unix::prelude::OwnedFd> {
        // todo!()
        None
    }
}
