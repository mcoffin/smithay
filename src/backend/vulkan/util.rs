//! General utilites and abstraction for dealing with the vulkan graphics API
//!
//! Most interesting in here are shortcuts that make building RAII-style abstractions over vulkan
//! types slightly easier (see [`OwnedHandle`] / [`ContextualHandle`]
use std::{
    fmt,
    ops::Deref,
    sync::{Arc, Weak},
};
use tracing::error;

pub(crate) trait VulkanHandle: Sized + Copy + Eq {
    fn null() -> Self;
    #[inline]
    fn is_null(&self) -> bool {
        self == &Self::null()
    }
}

pub(crate) trait ContextualHandle: VulkanHandle + fmt::Debug {
    type Context;
    const TYPE_NAME: &'static str;
    unsafe fn destroy(self, ctx: &Self::Context);
}

#[derive(Debug)]
pub(crate) struct OwnedHandle<T: ContextualHandle<Context = C>, C: fmt::Debug> {
    handle: T,
    context: Weak<C>,
}

impl<T, C> OwnedHandle<T, C>
where
    T: ContextualHandle<Context = C>,
    C: fmt::Debug,
{
    /// # Safety
    ///
    /// * `handle` must be valid for the lifetime of the object
    #[inline(always)]
    pub const unsafe fn new(handle: T, context: Weak<C>) -> Self {
        OwnedHandle { handle, context }
    }

    #[inline(always)]
    pub unsafe fn from_arc(handle: T, context: &Arc<C>) -> Self {
        OwnedHandle {
            handle,
            context: Arc::downgrade(context),
        }
    }

    pub fn into_raw(mut self) -> T {
        let mut tmp = T::null();
        std::mem::swap(&mut self.handle, &mut tmp);
        tmp
    }
}

impl<T, C> Deref for OwnedHandle<T, C>
where
    T: ContextualHandle<Context = C>,
    C: fmt::Debug,
{
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl<T, C> Drop for OwnedHandle<T, C>
where
    T: ContextualHandle<Context = C>,
    C: fmt::Debug,
{
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let Some(ctx) = self.context.upgrade() else {
                error!("destroyed after context: {:?}", self);
                return;
            };
            unsafe {
                self.handle.destroy(&ctx);
            }
        }
    }
}

impl<T, C> PartialEq<T> for OwnedHandle<T, C>
where
    T: ContextualHandle<Context = C> + Eq,
    C: fmt::Debug,
{
    #[inline(always)]
    fn eq(&self, rhs: &T) -> bool {
        &self.handle == rhs
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! vulkan_handles {
    ($($t:ty),+ $(,)?) => {
        $(
            impl $crate::backend::vulkan::util::VulkanHandle for $t {
                #[inline(always)]
                fn null() -> Self {
                    Self::null()
                }
            }
        )+
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! contextual_handles {
    ($ctx:ty { $($handle:ty = $destroy_fn:ident),+ $(,)? }) => {
        vulkan_handles!($($handle),+);
        $(
            impl $crate::backend::vulkan::util::ContextualHandle for $handle {
                type Context = $ctx;
                const TYPE_NAME: &'static str = stringify!($handle);
                #[inline(always)]
                unsafe fn destroy(self, ctx: &Self::Context) {
                    ctx.$destroy_fn(self, None);
                }
            }
        )+
    };
}
