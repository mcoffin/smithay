use ash::vk;
use crate::{
    vk_call,
    backend::renderer::sync::{Fence, SyncPoint},
};
use tracing::*;
use super::{
    Error,
    ErrorExt,
    SubmittedFrame,
};
use std::{
    fmt,
    mem::ManuallyDrop,
    sync::mpsc,
};

#[derive(Debug)]
pub struct VulkanFence {
    frame: ManuallyDrop<SubmittedFrame>,
    tx: mpsc::Sender<SubmittedFrame>,
}

impl VulkanFence {
    #[inline(always)]
    pub fn new(frame: SubmittedFrame, tx: mpsc::Sender<SubmittedFrame>) -> Self {
        VulkanFence {
            frame: ManuallyDrop::new(frame),
            tx,
        }
    }

    #[inline(always)]
    pub fn handle(&self) -> vk::Fence {
        self.frame.fence()
    }

    #[inline]
    pub fn status(&self) -> Result<FenceStatus, vk::Result> {
        self.frame.status()
    }
}

impl Drop for VulkanFence {
    fn drop(&mut self) {
        unsafe {
            let f = ManuallyDrop::take(&mut self.frame);
            self.tx.send(f)
                .expect("renderer did not exist to receive submitted frame info")
        }
    }
}

impl Fence for VulkanFence {
    fn is_signaled(&self) -> bool {
        match self.status() {
            Ok(FenceStatus::Signaled) => true,
            Ok(FenceStatus::Unsignaled) => false,
            Err(error) => {
                error!(?error, "error getting vulkan fence status");
                false
            },
        }
    }

    fn wait(&self) -> Result<(), crate::backend::renderer::sync::Interrupted> {
        use crate::backend::renderer::sync::Interrupted;
        self.frame.wait(u64::MAX)
            .map_err(|_| Interrupted)
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

    fn fence_status(
        &self,
        fence: vk::Fence,
    ) -> Result<FenceStatus, vk::Result>;

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

    #[inline(always)]
    fn fence_status(
        &self,
        fence: vk::Fence,
    ) -> Result<FenceStatus, vk::Result> {
        unsafe { self.get_fence_status(fence) }
            .map(|ret| if ret {
                FenceStatus::Signaled
            } else {
                FenceStatus::Unsignaled
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceStatus {
    Signaled,
    Unsignaled,
}

pub struct ScopeFence<'a> {
    handle: vk::Fence,
    device: &'a ash::Device,
}

impl<'a> ScopeFence<'a> {
    #[allow(dead_code)]
    #[inline(always)]
    pub fn handle(&self) -> vk::Fence {
        self.handle
    }
}

impl<'a> TryFrom<&'a ash::Device> for ScopeFence<'a> {
    type Error = Error<'static>;
    fn try_from(device: &'a ash::Device) -> Result<Self, Self::Error> {
        const INFO: vk::FenceCreateInfo = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: vk::FenceCreateFlags::empty(),
        };
        vk_call!(device, create_fence(&INFO, None)).map(|v| ScopeFence {
            handle: v,
            device,
        })
    }
}

impl<'a> fmt::Debug for ScopeFence<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopeFence")
            .field("handle", &self.handle)
            .field("device", &self.device.handle())
            .finish()
    }
}

impl<'a> Drop for ScopeFence<'a> {
    fn drop(&mut self) {
        if self.handle != vk::Fence::null() {
            if let Err(error) = vk_call!(self.device, wait_for_fences(&[self.handle], true, u64::MAX)) {
                error!(?error, "error waiting on fence");
            }
            unsafe { self.device.destroy_fence(self.handle, None); }
        }
    }
}
