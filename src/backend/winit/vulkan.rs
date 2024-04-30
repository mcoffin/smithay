use ash::vk;
use super::*;
use super::Error as WinitError;
use crate::backend::{
    renderer::vulkan::{
        Error as VkRendererError,
        VulkanRenderer,
        ErrorExt,
    },
    vulkan::{
        Instance,
        InstanceError,
        PhysicalDevice,
        Surface as VulkanSurface,
    },
};
use std::{
    ffi::CStr, mem::drop, rc::Rc
};

type WinitVulkanBackend = WinitGraphicsBackend<WinitVulkanGraphics>;

/// Create a new [`WinitVulkanBackend`]
///
/// See [`super::init`]
#[inline]
pub fn init() -> Result<(WinitVulkanBackend, WinitEventLoop), WinitError> {
    init_with_builder(builder())
}

/// Create a new [`WinitVulkanBackend`]
///
/// See [`super::init_with_builder`]
pub fn init_with_builder(builder: WindowBuilder) -> Result<(WinitVulkanBackend, WinitEventLoop), WinitError> {
    let span = info_span!("backend_winit", window = tracing::field::Empty);
    let _guard = span.enter();
    let (window, event_loop) = create_window(builder)?;
    span.record("window", Into::<u64>::into(window.id()));
    debug!("window created");

    let vk_gfx = WinitVulkanGraphics::with_window(&window)?;

    drop(_guard);

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let event_loop = Generic::new(event_loop, Interest::READ, calloop::Mode::Level);
    Ok((
        WinitGraphicsBackend {
            window: window.clone(),
            graphics: vk_gfx,
            bound_size: None,
            damage_tracking: false,
            span: span.clone(),
        },
        WinitEventLoop::new(
            window,
            event_loop,
            span
        )
    ))
}

/// [`WinitGraphics`] implementation details for a [`VulkanRenderer`]
#[derive(Debug)]
pub struct WinitVulkanGraphics {
    surface: Rc<VulkanSurface>,
    renderer: VulkanRenderer,
    physical_device: PhysicalDevice,
}

impl WinitVulkanGraphics {
    #[inline(always)]
    fn new(pd: PhysicalDevice, surface: VulkanSurface) -> Result<Self, Error> {
        VulkanRenderer::new(&pd)
            .map(move |renderer| WinitVulkanGraphics {
                renderer,
                physical_device: pd,
                surface: Rc::new(surface),
            })
            .map_err(Into::into)
    }

    /// Creates a new [`WinitVulkanGraphics`] based on a given window created by `winit`
    pub(super) fn with_window(window: &WinitWindow) -> Result<Self, Error> {
        let (instance, surface) = Instance::with_window(None, window)?;

        // TODO: use drm node or something to make better choices here
        let pd = instance.find_physical_device(
            &[],
            |pd| pd.properties().device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        ).or_else(|_| {
            instance.find_physical_device(&[], |_| true)
        })?;

        Self::new(pd, surface)
    }

    /// Gets a reference to the [`PhysicalDevice`] instance in use by this backend
    #[inline(always)]
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }
}

impl WinitGraphics for WinitVulkanGraphics {
    type Renderer = VulkanRenderer;
    type Surface = Rc<VulkanSurface>;

    #[inline(always)]
    fn surface(&self) -> &Self::Surface {
        &self.surface
    }

    /// This is a no-op on winit+vulkan as the submission is handled in [`Frame::finish`]
    #[inline(always)]
    fn submit(
        &mut self,
        _damage: Option<&mut [Rectangle<i32, Physical>]>,
    ) -> Result<(), crate::backend::SwapBuffersError> {
        Ok(())
    }
}

impl Drop for WinitVulkanGraphics {
    #[tracing::instrument(skip(self), name = "winit_vulkan_destroy")]
    fn drop(&mut self) {
        use crate::backend::renderer::Unbind;
        if let Err(error) = self.renderer.unbind() {
            error!(?error, "failed to unbind renderer");
        }
    }
}

impl WinitSurface for Rc<VulkanSurface> {
    type Error = VkRendererError<'static>;

    fn resize(
        &self,
        width: i32,
        height: i32,
        _dx: i32,
        _dy: i32
    ) -> Result<(), Self::Error> {
        VulkanSurface::resize(self, width as _, height as _);
        Ok(())
    }
}

macro_rules! as_ref_mut {
    ($t:ty { $($field:ident : $field_ty:ty),+ $(,)? }) => {
        $(
            impl AsRef<$field_ty> for $t {
                #[inline(always)]
                fn as_ref(&self) -> &$field_ty {
                    &self.$field
                }
            }

            impl AsMut<$field_ty> for $t {
                fn as_mut(&mut self) -> &mut $field_ty {
                    &mut self.$field
                }
            }
        )+
    };
}

as_ref_mut!(WinitVulkanGraphics {
    renderer: VulkanRenderer,
});

trait InstanceExt {
    fn find_physical_device<F>(
        &self,
        required_extensions: &[&CStr],
        predicate: F
    ) -> Result<PhysicalDevice, Error>
    where
        F: Fn(&PhysicalDevice) -> bool;
}

impl InstanceExt for Instance {
    fn find_physical_device<F>(
        &self,
        required_extensions: &[&CStr],
        predicate: F
    ) -> Result<PhysicalDevice, Error>
    where
        F: Fn(&PhysicalDevice) -> bool,
    {
        PhysicalDevice::enumerate(self)
            .vk("vkEnumeratePhysicalDevices")?
            .filter(|pd| {
                !required_extensions.iter()
                    .copied()
                    .any(|name| !pd.has_device_extension(name))
            })
            .find(predicate)
            .ok_or(Error::NoSuitableDevice)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("error creating vulkan instance: {0}")]
    Instance(#[from] InstanceError),
    #[error("error setting up vulkan context: {0}")]
    Renderer(#[from] VkRendererError<'static>),
    #[error("no suitable physical device found")]
    NoSuitableDevice,
}
