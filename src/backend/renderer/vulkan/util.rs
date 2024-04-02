use ash::{
    vk,
    extensions::khr::ExternalMemoryFd,
};
use crate::backend::allocator::dmabuf::Dmabuf;
use std::{
    fmt,
    ops::Deref,
    os::fd::{
        BorrowedFd,
        AsRawFd,
    },
};
use super::MAX_PLANES;

#[derive(Clone, Copy)]
pub struct DerefSliceIter<T> {
    arr: T,
    idx: usize,
}

impl<T> DerefSliceIter<T> {
    pub const fn new(arr: T) -> Self {
        DerefSliceIter { arr, idx: 0 }
    }
}

impl<T, V> Iterator for DerefSliceIter<T>
where
    V: Clone,
    T: Deref<Target = [V]>,
{
    type Item = V;
    fn next(&mut self) -> Option<V> {
        if self.idx >= self.arr.len() {
            None
        } else {
            let ret = self.arr[self.idx].clone();
            self.idx += 1;
            Some(ret)
        }
    }
}

impl<T, V> ExactSizeIterator for DerefSliceIter<T>
where
    V: Clone,
    T: Deref<Target = [V]>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.arr.len()
    }
}

#[derive(Clone, Copy)]
pub struct DmabufMemoryProperties<'a> {
    props: [(BorrowedFd<'a>, vk::MemoryFdPropertiesKHR); MAX_PLANES],
    props_len: usize,
}

impl<'a> DmabufMemoryProperties<'a> {
    #[inline(always)]
    pub fn as_slice(&self) -> &[(BorrowedFd<'a>, vk::MemoryFdPropertiesKHR)] {
        &self.props[..self.props_len]
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.props_len
    }
}

impl<'a> AsRef<[(BorrowedFd<'a>, vk::MemoryFdPropertiesKHR)]> for DmabufMemoryProperties<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[(BorrowedFd<'a>, vk::MemoryFdPropertiesKHR)] {
        self.as_slice()
    }
}

impl<'a> IntoIterator for DmabufMemoryProperties<'a> {
    type Item = (BorrowedFd<'a>, vk::MemoryFdPropertiesKHR);
    type IntoIter = std::iter::Take<std::array::IntoIter<(BorrowedFd<'a>, vk::MemoryFdPropertiesKHR), 4>>;

    fn into_iter(self) -> Self::IntoIter {
        self.props.into_iter()
            .take(std::cmp::min(self.len(), self.props.len()))
    }
}

impl<'a> fmt::Debug for DmabufMemoryProperties<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_slice())
    }
}

pub trait ExternalMemoryFdExt {
    fn dmabuf_memory_properties<'a>(&self, buffer: &'a Dmabuf) -> Result<DmabufMemoryProperties<'a>, vk::Result>;
}
impl ExternalMemoryFdExt for ExternalMemoryFd {
    fn dmabuf_memory_properties<'a>(&self, buffer: &'a Dmabuf) -> Result<DmabufMemoryProperties<'a>, vk::Result> {
        const HTYPE: vk::ExternalMemoryHandleTypeFlags =
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let default_value = unsafe {
            (BorrowedFd::borrow_raw(0), vk::MemoryFdPropertiesKHR::default())
        };
        let mut ret = [default_value; MAX_PLANES];
        let mut count = 0usize;
        for (fd, out) in buffer.handles().take(MAX_PLANES).zip(ret.iter_mut()) {
            let props = unsafe {
                self.get_memory_fd_properties(HTYPE, fd.as_raw_fd())
            }?;
            *out = (fd, props);
            count += 1;
        }
        Ok(DmabufMemoryProperties {
            props: ret,
            props_len: count,
        })
    }
}
