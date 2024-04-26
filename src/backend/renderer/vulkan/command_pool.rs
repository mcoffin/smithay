use ash::vk;
use std::{
    fmt,
    iter,
    marker::PhantomData,
    ops::{
        Deref,
        DerefMut,
    },
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            Ordering,
        },
    },
};
use scopeguard::ScopeGuard;
use super::{
    Error,
    ErrorExt,
};

pub struct CommandBufferPool {
    device: Arc<super::Device>,
    pool: Pool<vk::Semaphore, vk::Result>,
    command_pool: vk::CommandPool,
    command_buffers: Box<[vk::CommandBuffer]>,
}

impl CommandBufferPool {
    pub fn new(device: Arc<super::Device>, n: usize, pool_info: &vk::CommandPoolCreateInfo) -> Result<Self, Error<'static>> {
        let cmd_pool = unsafe {
            device.create_command_pool(&pool_info, None)
        }.vk("vkCreateCommandPool").map(|v| scopeguard::guard(v, |v| unsafe {
            device.destroy_command_pool(v, None);
        }))?;
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(*cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(n as _);
        let buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .vk("vkAllocateCommandBuffers")?;
        let idevice = device.clone();
        let ddevice = device.clone();
        let pool = Pool::new(
            move || unsafe {
                let info = vk::SemaphoreCreateInfo::builder()
                    .flags(vk::SemaphoreCreateFlags::empty())
                    .build();
                idevice.create_semaphore(&info, None)
            },
            move |entries| unsafe {
                let handles = entries.iter()
                    .filter_map(|&v| v)
                    .filter(|handle| handle != &vk::Semaphore::null());
                for h in handles {
                    ddevice.destroy_semaphore(h, None);
                }
            },
            n
        );
        Ok(CommandBufferPool {
            pool,
            command_pool: ScopeGuard::into_inner(cmd_pool),
            command_buffers: buffers.into(),
            device,
        })
    }

    pub fn acquire(&mut self) -> Result<PooledCommandBuffer<'_>, AcquireError<vk::Result>> {
        let entry = self.pool.acquire()?;
        Ok(PooledCommandBuffer {
            buffer: self.command_buffers[entry.idx],
            acquire_semaphore: entry,
            _marker: PhantomData,
        })
    }
}

impl Drop for CommandBufferPool {
    fn drop(&mut self) {
        unsafe {
            self.device.free_command_buffers(self.command_pool, &self.command_buffers);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[derive(Debug)]
pub struct PooledCommandBuffer<'a> {
    buffer: vk::CommandBuffer,
    acquire_semaphore: Entry<vk::Semaphore>,
    _marker: PhantomData<&'a CommandBufferPool>,
}

impl<'a> PooledCommandBuffer<'a> {
    #[inline(always)]
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.buffer
    }
    #[inline(always)]
    pub fn acquire_semaphore(&self) -> vk::Semaphore {
        *self.acquire_semaphore
    }
}

struct Pool<T, E> {
    entries: Box<[Option<T>]>,
    usage: Box<[Arc<AtomicBool>]>,
    init_fn: Box<dyn Fn() -> Result<T, E>>,
    destroy_fn: Box<dyn Fn(&mut [Option<T>])>,
}

impl<T: Clone + Copy, E> Pool<T, E> {
    fn new<InitF, DestroyF>(init_fn: InitF, destroy_fn: DestroyF, n: usize) -> Self
    where
        InitF: Fn() -> Result<T, E>,
        InitF: Sized + 'static,
        DestroyF: Fn(&mut [Option<T>]),
        DestroyF: Sized + 'static,
    {
        let mut entries = Vec::with_capacity(n);
        entries.extend(iter::repeat(None).take(n));
        let mut usage = Vec::with_capacity(n);
        usage.extend(iter::repeat_with(|| Arc::new(AtomicBool::new(false))).take(n));
        Pool {
            entries: entries.into(),
            usage: usage.into(),
            init_fn: Box::new(init_fn) as _,
            destroy_fn: Box::new(destroy_fn) as _,
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn acquire(&mut self) -> Result<Entry<T>, AcquireError<E>> {
        enum Found<'a, FT> {
            Uninitialized(&'a mut FT, Arc<AtomicBool>),
            Unused(&'a FT, Arc<AtomicBool>),
        }
        let (idx, slot, usage) = self.entries.iter_mut()
            .zip(&*self.usage)
            .enumerate()
            .find_map(|(idx, (item, usage))| {
                if usage.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst) == Ok(false) {
                    Some((idx, item, usage.clone()))
                } else {
                    None
                }
            })
            .ok_or(AcquireError::Availability)?;
        if let &mut Some(v) = slot {
            Ok(Entry {
                inner: v,
                busy: usage,
                idx,
            })
        } else {
            let v = (self.init_fn)()?;
            *slot = Some(v);
            Ok(Entry {
                inner: v,
                busy: usage,
                idx,
            })
        }
    }
}

impl<T, E> fmt::Debug for Pool<T, E>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pool")
            .field("entries", &&*self.entries)
            .field("usage", &&*self.usage)
            .finish()
    }
}

impl<T, E> Drop for Pool<T, E> {
    fn drop(&mut self) {
        (self.destroy_fn)(&mut self.entries);
    }
}

#[derive(Debug, Clone)]
pub struct Entry<T> {
    inner: T,
    busy: Arc<AtomicBool>,
    idx: usize,
}

impl<T> Deref for Entry<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.inner
    }
}
impl<T> DerefMut for Entry<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T> Drop for Entry<T> {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::SeqCst);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AcquireError<E> {
    #[error("no available items left in pool")]
    Availability,
    #[error("error initializing pool item: {0}")]
    Creation(#[from] E),
}
