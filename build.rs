#![allow(clippy::disallowed_macros)]
#[cfg(any(feature = "backend_egl", feature = "renderer_gl"))]
fn gl_generate() {
    use gl_generator::{Api, Fallbacks, Profile, Registry};
    use std::{env, fs::File, path::PathBuf};

    let dest = PathBuf::from(&env::var("OUT_DIR").unwrap());

    if env::var_os("CARGO_FEATURE_BACKEND_EGL").is_some() {
        let mut file = File::create(dest.join("egl_bindings.rs")).unwrap();
        Registry::new(
            Api::Egl,
            (1, 5),
            Profile::Core,
            Fallbacks::All,
            [
                "EGL_KHR_create_context",
                "EGL_EXT_create_context_robustness",
                "EGL_KHR_create_context_no_error",
                "EGL_KHR_no_config_context",
                "EGL_EXT_pixel_format_float",
                "EGL_EXT_device_base",
                "EGL_EXT_device_enumeration",
                "EGL_EXT_device_query",
                "EGL_EXT_device_drm",
                "EGL_EXT_device_drm_render_node",
                "EGL_KHR_stream",
                "EGL_KHR_stream_producer_eglsurface",
                "EGL_EXT_platform_base",
                "EGL_KHR_platform_x11",
                "EGL_EXT_platform_x11",
                "EGL_KHR_platform_wayland",
                "EGL_EXT_platform_wayland",
                "EGL_KHR_platform_gbm",
                "EGL_MESA_platform_gbm",
                "EGL_MESA_platform_surfaceless",
                "EGL_EXT_platform_device",
                "EGL_WL_bind_wayland_display",
                "EGL_KHR_image_base",
                "EGL_EXT_image_dma_buf_import",
                "EGL_EXT_image_dma_buf_import_modifiers",
                "EGL_MESA_image_dma_buf_export",
                "EGL_KHR_gl_image",
                "EGL_EXT_buffer_age",
                "EGL_EXT_swap_buffers_with_damage",
                "EGL_KHR_swap_buffers_with_damage",
                "EGL_KHR_fence_sync",
                "EGL_ANDROID_native_fence_sync",
                "EGL_IMG_context_priority",
            ],
        )
        .write_bindings(gl_generator::GlobalGenerator, &mut file)
        .unwrap();
    }

    if env::var_os("CARGO_FEATURE_RENDERER_GL").is_some() {
        let mut file = File::create(dest.join("gl_bindings.rs")).unwrap();
        Registry::new(
            Api::Gles2,
            (3, 2),
            Profile::Compatibility,
            Fallbacks::None,
            [
                "GL_OES_EGL_image",
                "GL_OES_EGL_image_external",
                "GL_EXT_texture_format_BGRA8888",
                "GL_EXT_unpack_subimage",
                "GL_OES_EGL_sync",
            ],
        )
        .write_bindings(gl_generator::StructGenerator, &mut file)
        .unwrap();
    }
}

#[cfg(all(feature = "backend_gbm", not(feature = "backend_gbm_has_fd_for_plane")))]
fn test_gbm_bo_fd_for_plane() {
    let gbm = match pkg_config::probe_library("gbm") {
        Ok(lib) => lib,
        Err(_) => {
            println!("cargo:warning=failed to find gbm, assuming gbm_bo_get_fd_for_plane is unavailable");
            return;
        }
    };

    let has_gbm_bo_get_fd_for_plane = cc::Build::new()
        .file("test_gbm_bo_get_fd_for_plane.c")
        .includes(gbm.include_paths)
        .warnings_into_errors(true)
        .cargo_metadata(false)
        .try_compile("test_gbm_bo_get_fd_for_plane")
        .is_ok();

    if has_gbm_bo_get_fd_for_plane {
        println!("cargo:rustc-cfg=feature=\"backend_gbm_has_fd_for_plane\"");
    }
}

#[cfg(all(
    feature = "backend_gbm",
    not(feature = "backend_gbm_has_create_with_modifiers2")
))]
fn test_gbm_bo_create_with_modifiers2() {
    let gbm = match pkg_config::probe_library("gbm") {
        Ok(lib) => lib,
        Err(_) => {
            println!(
                "cargo:warning=failed to find gbm, assuming gbm_bo_create_with_modifiers2 is unavailable"
            );
            return;
        }
    };

    let has_gbm_bo_create_with_modifiers2 = cc::Build::new()
        .file("test_gbm_bo_create_with_modifiers2.c")
        .includes(gbm.include_paths)
        .warnings_into_errors(true)
        .cargo_metadata(false)
        .try_compile("test_gbm_bo_create_with_modifiers2")
        .is_ok();

    if has_gbm_bo_create_with_modifiers2 {
        println!("cargo:rustc-cfg=feature=\"backend_gbm_has_create_with_modifiers2\"");
    }
}

#[cfg(feature = "renderer_vulkan")]
fn vk_compile_shaders() -> Result<(), Box<dyn std::error::Error>> {
    use std::{
        env,
        io,
        process::{
            Command,
            ExitStatus,
        },
        path::{
            Path,
            PathBuf,
        },
        ffi::OsString,
    };
    #[derive(Debug, thiserror::Error)]
    #[error("missing required environment variable: {0}")]
    struct MissingEnv<'a>(&'a str);

    #[inline]
    fn required_env(key: &str) -> Result<OsString, MissingEnv<'_>> {
        env::var_os(key).ok_or(MissingEnv(key))
    }

    #[derive(Debug, thiserror::Error)]
    enum CommandError {
        #[error("error running command: {0}")]
        Io(#[from] io::Error),
        #[error("command exited with failure status: {0:?}")]
        Exit(ExitStatus),
    }

    #[derive(Debug, thiserror::Error)]
    #[error("error running {command}: {error}")]
    struct NamedCommandError<'a> {
        command: &'a str,
        error: CommandError,
    }

    impl CommandError {
        #[inline(always)]
        fn named(self, name: &str) -> NamedCommandError<'_> {
            NamedCommandError {
                command: name,
                error: self,
            }
        }
    }

    trait CommandExt {
        fn status_checked(&mut self) -> Result<(), CommandError>;
    }
    impl CommandExt for Command {
        fn status_checked(&mut self) -> Result<(), CommandError> {
            self.status()
                .map_err(CommandError::from)
                .and_then(|status| if status.success() {
                    Ok(())
                } else {
                    Err(CommandError::Exit(status))
                })
        }
    }

    let out_dir = required_env("OUT_DIR")
        .map(PathBuf::from)?;
    let compile_shader = |in_path: &str, out_name: &str| -> Result<(), NamedCommandError<'static>> {
        const GLSLC: &str = "glslc";
        let out_path = out_dir.join(out_name);
        println!("cargo:rerun-if-changed={}", in_path);
        Command::new(GLSLC)
            .args(["--target-env=vulkan1.1", in_path, "-o"])
            .arg(out_path)
            .status_checked()
            .map_err(|e| e.named(GLSLC))
    };
    macro_rules! compile_shaders {
        ($($name:expr),+ $(,)?) => {
            $(
                compile_shader(
                    concat!("src/backend/renderer/vulkan/shaders/", $name),
                    concat!($name, ".spv"),
                )?;
            )+
        };
    }
    compile_shaders! {
        "common.vert",
        "quad.frag",
    }
    Ok(())
}

fn main() {
    #[cfg(any(feature = "backend_egl", feature = "renderer_gl"))]
    gl_generate();

    #[cfg(all(feature = "backend_gbm", not(feature = "backend_gbm_has_fd_for_plane")))]
    test_gbm_bo_fd_for_plane();
    #[cfg(all(
        feature = "backend_gbm",
        not(feature = "backend_gbm_has_create_with_modifiers2")
    ))]
    test_gbm_bo_create_with_modifiers2();

    #[cfg(feature = "renderer_vulkan")]
    vk_compile_shaders().expect("failed to compile vulkan shaders");
}
