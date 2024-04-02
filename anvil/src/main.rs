#[derive(Debug, clap::Parser)]
struct Args {
    #[arg(long, short)]
    renderer: Renderer,
    #[arg(required = true)]
    backend: Backend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Backend {
    #[value(name = "winit")]
    Winit,
    #[value(name = "udev")]
    Udev,
    #[value(name = "x11")]
    X11,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Renderer {
    #[value(name = "gles")]
    Gles,
    #[value(name = "vulkan")]
    Vulkan,
}

impl Default for Renderer {
    #[inline(always)]
    fn default() -> Self {
        Renderer::Gles
    }
}

#[cfg(feature = "profile-with-tracy-mem")]
#[global_allocator]
static GLOBAL: profiling::tracy_client::ProfiledAllocator<std::alloc::System> =
    profiling::tracy_client::ProfiledAllocator::new(std::alloc::System, 10);

fn main() {
    use clap::Parser;
    if let Ok(env_filter) = tracing_subscriber::EnvFilter::try_from_default_env() {
        tracing_subscriber::fmt()
            .compact()
            .with_env_filter(env_filter)
            .init();
    } else {
        tracing_subscriber::fmt().compact().init();
    }

    #[cfg(feature = "profile-with-tracy")]
    profiling::tracy_client::Client::start();

    profiling::register_thread!("Main Thread");

    #[cfg(feature = "profile-with-puffin")]
    let _server = puffin_http::Server::new(&format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT)).unwrap();
    #[cfg(feature = "profile-with-puffin")]
    profiling::puffin::set_scopes_on(true);

    let config = Args::parse();
    match (config.backend, config.renderer) {
        (Backend::Winit, Renderer::Gles) => {
            tracing::info!("Starting anvil with winit/gles backend");
            anvil::winit::run_winit_gles();
        }
        #[cfg(feature = "renderer_vulkan")]
        (Backend::Winit, Renderer::Vulkan) => {
            tracing::info!("Starting anvil with winit/vulkan backend");
            anvil::winit::run_winit_vulkan();
        }
        (Backend::Udev, ..) => {
            tracing::info!("Starting anvil on a tty using udev");
            anvil::udev::run_udev();
        }
        (Backend::X11, ..) => {
            tracing::info!("Starting anvil with x11 backend");
            anvil::x11::run_x11();
        }
    }
}
