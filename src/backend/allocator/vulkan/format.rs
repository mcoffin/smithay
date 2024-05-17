//! Format conversions between Vulkan and DRM formats.

use crate::backend::allocator::Fourcc;
use ash::vk;
use core::num::NonZeroI32;
use std::fmt;

macro_rules! vk_format_mapping {
    ($vk:ident) => {
        FormatMapping { format: ash::vk::Format::$vk, format_srgb: None }
    };
    ($vk:ident, $vk_srgb:ident) => {
        FormatMapping::new(ash::vk::Format::$vk, ash::vk::Format::$vk_srgb)
    };
}

/// Macro to generate format conversions between Vulkan and FourCC format codes.
///
/// Any entry in this table may have attributes associated with a conversion. This is needed for `PACK` Vulkan
/// formats which may only have an alternative given a specific host endian.
///
/// See the module documentation for usage details.
macro_rules! vk_format_table {
    (
        $(
            // This meta specifier is used for format conversions for PACK formats.
            $(#[$conv_meta:meta])*
            $fourcc: ident => $vk: ident $(| $vk_srgb:ident)?
        ),* $(,)?
    ) => {
        /// Converts a FourCC format code to a Vulkan format code.
        ///
        /// This will return [`None`] if the format is not known.
        ///
        /// These format conversions will return all known FourCC and Vulkan format conversions. However a
        /// Vulkan implementation may not support some Vulkan format. One notable example of this are the
        /// formats introduced in `VK_EXT_4444_formats`. The corresponding FourCC codes will return the
        /// formats from `VK_EXT_4444_formats`, but the caller is responsible for testing that a Vulkan device
        /// supports these formats.
        pub const fn get_vk_format(fourcc: $crate::backend::allocator::Fourcc) -> Option<ash::vk::Format> {
            // FIXME: Use reexport for ash::vk::Format
            match fourcc {
                $(
                    $(#[$conv_meta])*
                    $crate::backend::allocator::Fourcc::$fourcc => Some(ash::vk::Format::$vk),
                )*

                _ => None,
            }
        }

        /// Returns all the known format conversions.
        ///
        /// The list contains FourCC format codes that may be converted using [`get_vk_format`].
        pub const fn known_formats() -> &'static [$crate::backend::allocator::Fourcc] {
            &[
                $(
                    $crate::backend::allocator::Fourcc::$fourcc
                ),*
            ]
        }

        pub const FORMAT_MAPPINGS: &[(Fourcc, FormatMapping)] = &[
            $(
                $(#[$conv_meta])*
                ($crate::backend::allocator::Fourcc::$fourcc, vk_format_mapping!($vk $(, $vk_srgb)?)),
            )*
        ];
    };
}

#[derive(Hash, Clone, Copy, PartialEq, Eq)]
pub struct FormatMapping {
    pub format: vk::Format,
    format_srgb: Option<NonZeroI32>,
}

impl FormatMapping {
    const fn new(format: vk::Format, format_srgb: vk::Format) -> Self {
        FormatMapping {
            format,
            format_srgb: NonZeroI32::new(format_srgb.as_raw()),
        }
    }
    #[inline(always)]
    pub const fn has_srgb(&self) -> bool {
        self.format_srgb.is_some()
    }
    #[inline(always)]
    pub const fn srgb(&self) -> Option<vk::Format> {
        match self.format_srgb {
            Some(v) => Some(vk::Format::from_raw(v.get())),
            None => None
        }
    }
}

impl From<vk::Format> for FormatMapping {
    #[inline(always)]
    fn from(format: vk::Format) -> Self {
        FormatMapping {
            format,
            format_srgb: None,
        }
    }
}

impl fmt::Debug for FormatMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FormatMapping")
            .field("format", &self.format)
            .field("format_srgb", &self.srgb())
            .finish()
    }
}

// FIXME: SRGB format is not always correct.
//
// Vulkan classifies formats by both channel sizes and colorspace. FourCC format codes do not classify formats
// based on colorspace.
//
// To implement this correctly, it is likely that parsing vulkan.xml and classifying families of colorspaces
// would be needed since there are a lot of formats.
//
// Many of these conversions come from wsi_common_wayland.c in Mesa
vk_format_table! {
    Argb8888 => B8G8R8A8_UNORM,
    Xrgb8888 => B8G8R8A8_UNORM | B8G8R8A8_SRGB,

    Abgr8888 => R8G8B8A8_UNORM,
    Xbgr8888 => R8G8B8A8_UNORM | R8G8B8A8_SRGB,

    Rgb888 => B8G8R8_UNORM | B8G8R8_SRGB,
    Bgr888 => R8G8B8_UNORM | R8G8B8_SRGB,

    // PACK32 formats are equivalent to u32 instead of [u8; 4] and thus depend their layout depends the host
    // endian.
    #[cfg(target_endian = "little")]
    Rgba8888 => A8B8G8R8_SRGB_PACK32,
    #[cfg(target_endian = "little")]
    Rgbx8888 => A8B8G8R8_SRGB_PACK32,

    #[cfg(target_endian = "little")]
    Argb2101010 => A2R10G10B10_UNORM_PACK32,
    #[cfg(target_endian = "little")]
    Xrgb2101010 => A2R10G10B10_UNORM_PACK32,

    #[cfg(target_endian = "little")]
    Abgr2101010 => A2B10G10R10_UNORM_PACK32,
    #[cfg(target_endian = "little")]
    Xbgr2101010 => A2B10G10R10_UNORM_PACK32,

    Abgr16161616f => R16G16B16A16_SFLOAT,
    Xbgr16161616f => R16G16B16A16_SFLOAT,
}

pub(crate) fn known_vk_formats() -> impl Iterator<Item = (Fourcc, FormatMapping)> {
    FORMAT_MAPPINGS.iter().copied()
}

pub(crate) trait FormatExt {
    fn has_alpha(&self) -> bool;
}

impl FormatExt for vk::Format {
    fn has_alpha(&self) -> bool {
        !matches!(
            *self,
            vk::Format::B8G8R8_SRGB
            | vk::Format::B8G8R8_UNORM
            | vk::Format::R8G8B8_SRGB
            | vk::Format::R8G8B8_UNORM
        )
    }
}

impl FormatExt for FormatMapping {
    fn has_alpha(&self) -> bool {
        self.format.has_alpha() || self.srgb().map_or(false, |f| f.has_alpha())
    }
}

// pub const MEM_FORMATS: &[(Fourcc, vk::Format)] = &[
//     (Fourcc::Argb8888, vk::Format::B8G8R8A8_UNORM),
//     (Fourcc::Xrgb8888, vk::Format::B8G8R8A8_SRGB),
//     (Fourcc::Abgr8888, vk::Format::R8G8B8A8_UNORM),
//     (Fourcc::Xbgr8888, vk::Format::R8G8B8A8_SRGB),
//     (Fourcc::Rgb888, vk::Format::B8G8R8_SRGB),
//     (Fourcc::Bgr888, vk::Format::R8G8B8_SRGB),
// ];
