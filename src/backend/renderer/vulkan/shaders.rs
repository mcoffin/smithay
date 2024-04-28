use ash::vk;
use core::ops::Deref;

macro_rules! shader_sources {
    ($name:ident, $ty:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".", $ty, ".spv"))
    };
    ($(($ty:ident, $sname:ident) { $($name:ident),+ $(,)? })+) => {
        pub struct ShaderSources<'a> {
            $(pub $ty : $sname<'a>),+
        }

        $(
            pub struct $sname<'a> {
                $(
                    pub $name : &'a [u8],
                )+
            }
        )+

        pub static SOURCES: ShaderSources<'static> = ShaderSources {
            $(
                $ty: $sname {
                    $(
                        $name : shader_sources!($name, stringify!($ty)),
                    )+
                },
            )+
        };
    };
}

shader_sources! {
    (vert, ShaderSourcesVert) {
        common,
    }
    (frag, ShaderSourcesFrag) {
        quad,
    }
}

pub struct ShaderModule<'a> {
    handle: vk::ShaderModule,
    device: &'a ash::Device,
}

impl<'a> ShaderModule<'a> {
    pub fn new(device: &'a ash::Device, source: &[u8]) -> Result<Self, vk::Result> {
        Self::with_flags(device, source, vk::ShaderModuleCreateFlags::empty())
    }

    pub fn with_flags(device: &'a ash::Device, source: &[u8], flags: vk::ShaderModuleCreateFlags) -> Result<Self, vk::Result> {
        use core::ptr;
        let create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags,
            code_size: source.len(),
            p_code: source.as_ptr().cast(),
        };
        unsafe {
            device.create_shader_module(&create_info, None)
        }.map(|handle| ShaderModule {
            handle,
            device,
        })
    }

    #[inline(always)]
    pub unsafe fn from_raw(device: &'a ash::Device, handle: vk::ShaderModule) -> Self {
        ShaderModule { handle, device }
    }

    #[inline(always)]
    pub fn handle(&self) -> vk::ShaderModule {
        self.handle
    }
}

impl<'a> Drop for ShaderModule<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.handle, None);
        }
    }
}

impl<'a> Deref for ShaderModule<'a> {
    type Target = vk::ShaderModule;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}
