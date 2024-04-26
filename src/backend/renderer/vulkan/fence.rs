use ash::vk;
use std::sync::Arc;
use crate::{
    backend::renderer::sync::{Fence, SyncPoint}, fn_name
};
use tracing::*;
use super::{
    Error,
    ErrorExt,
};
#[derive(Debug)]
pub struct VulkanFence {
    device: Arc<super::Device>,
    handle: vk::Fence,
    pub(super) image_ready: vk::Semaphore,
    pub(super) cb: CommandBuffer,
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
            image_ready: vk::Semaphore::null(),
            cb: Default::default(),
        })
    }

    #[inline(always)]
    pub fn handle(&self) -> vk::Fence {
        self.handle
    }

    #[inline]
    pub fn status(&self) -> Result<FenceStatus, vk::Result> {
        if unsafe { self.device.get_fence_status(self.handle) }? {
            Ok(FenceStatus::Signaled)
        } else {
            Ok(FenceStatus::Unsignaled)
        }
    }
}

impl Drop for VulkanFence {
    fn drop(&mut self) {
        match self.status() {
            Ok(FenceStatus::Unsignaled) => if let Err(error) = unsafe {
                self.device.wait_for_fences(&[self.handle], true, u64::MAX)
            } {
                error!(?error, "error waiting on fence in destructor");
            },
            Err(error) => {
                error!(?error, "error getting fence status");
            },
            _ => {},
        }
        unsafe {
            if self.cb.is_valid() {
                self.device.free_command_buffers(self.cb.pool, &self.cb.buffers);
            }
            if self.image_ready != vk::Semaphore::null() {
                self.device.destroy_semaphore(self.image_ready, None);
            }
            self.device.destroy_fence(self.handle, None);
        }
    }
}

impl Fence for VulkanFence {
    fn is_signaled(&self) -> bool {
        let ret = self.status();
        if let Err(error) = &ret {
            error!(?error, "{}", fn_name!());
        }
        match self.status() {
            Ok(FenceStatus::Signaled) => true,
            Err(..) => false,
            _ => false,
        }
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

pub trait DeviceExtFence {
    /// Slim wrapper around [`ash::Device::wait_for_fences`]
    fn wait_fences(
        &self,
        fences: &[vk::Fence],
        wait_all: bool,
        timeout: u64
    ) -> Result<(), Error<'static>>;

    /// Attempts to downcast this [`SyncPoint`] as a [`VulkanFence`], and preserve native results
    /// if so.
    ///
    /// If the [`SyncPoint`] is *not* a [`VulkanFence`], then just wrapp errors in
    /// [`Error::Interrupted`] after calling [`SyncPoint::wait`]
    fn wait_fence_vk(&self, fence: &SyncPoint) -> Result<(), Error<'static>> {
        if let Some(sync) = fence.get::<VulkanFence>() {
            self.wait_fences(&[sync.handle()], true, u64::MAX)
        } else {
            fence.wait().map_err(From::from)
        }
    }
}

impl DeviceExtFence for ash::Device {
    #[inline(always)]
    fn wait_fences(
        &self,
        fences: &[vk::Fence],
        wait_all: bool,
        timeout: u64
    ) -> Result<(), Error<'static>> {
        unsafe {
            self.wait_for_fences(fences, wait_all, timeout)
        }.vk("vkWaitOnFences")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CommandBuffer {
    buffers: [vk::CommandBuffer; 1],
    pool: vk::CommandPool,
}

impl CommandBuffer {
    #[inline(always)]
    pub fn new(buffer: vk::CommandBuffer, pool: vk::CommandPool) -> Self {
        CommandBuffer {
            buffers: [buffer],
            pool,
        }
    }

    fn is_valid(&self) -> bool {
        self.buffers[0] != vk::CommandBuffer::null() && self.pool != vk::CommandPool::null()
    }
}

impl Default for CommandBuffer {
    #[inline(always)]
    fn default() -> Self {
        CommandBuffer {
            buffers: [vk::CommandBuffer::null(); 1],
            pool: vk::CommandPool::null(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceStatus {
    Signaled,
    Unsignaled,
}
