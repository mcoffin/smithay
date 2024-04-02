use std::{
    io,
    os::fd::{
        BorrowedFd,
        AsRawFd,
        FromRawFd,
        IntoRawFd,
    },
};
use crate::backend::allocator::dmabuf::Dmabuf;

pub trait DmabufExt {
    /// Uses inode id to determine if all planes of a dmabuf belong to the same memory allocation
    fn is_disjoint(&self) -> Result<bool, DmabufError>;
}

impl DmabufExt for Dmabuf {
    fn is_disjoint(&self) -> Result<bool, DmabufError> {
        use std::{
            os::linux::fs::MetadataExt,
            fs,
        };
        /// Allows retrieval of [`fs::Metadata`] without having to duplicate a [`BorrowedFd`]
        trait BorrowedFdExt {
            fn metadata(&self) -> io::Result<fs::Metadata>;
        }
        impl<'a> BorrowedFdExt for BorrowedFd<'a> {
            fn metadata(&self) -> io::Result<fs::Metadata> {
                // Safe as long as we're *damn* sure that we re-take ownership during our borrow of the fd,
                // and before the [`fs::File`] is dropped
                let f = unsafe {
                    fs::File::from_raw_fd(self.as_raw_fd())
                };
                let ret = f.metadata();
                // since we don't *actually* own the file descriptor, take "ownership" back so that
                // dropping the file doesn't close the fd
                f.into_raw_fd();
                ret
            }
        }
        let mut it = self.handles();
        let first_fd = if let Some(v) = it.next() {
            v
        } else {
            return Err(DmabufError::NoPlanes);
        };
        let last_ino = first_fd.metadata()
            .map(|info| info.st_ino())?;
        for fd in it {
            let ino = fd.metadata()
                .map(|info| info.st_ino())?;
            if ino != last_ino {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DmabufError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("dmabuf contained no planes")]
    NoPlanes,
    #[error("no memory type found for bits: 0b{0:b}")]
    NoMemoryType(u32),
}
