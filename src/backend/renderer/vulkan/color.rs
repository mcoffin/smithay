#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct LinearColor(pub [f32; 4]);

/// Shamelessly ripped out of wlroots
fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb > 0.04045 {
        ((srgb + 0.055) / 1.055).powf(2.4)
    } else {
        srgb / 12.92
    }
}

impl LinearColor {
    pub fn from_srgb_premultiplied(color: &[f32; 4]) -> Self {
        let alpha = color[3];
        let mut ret = [0.0, 0.0, 0.0, alpha];
        if alpha != 0.0 {
            for (&v, out) in color[..3].iter().zip(&mut ret) {
                *out = srgb_to_linear(v / alpha) * alpha;
            }
        }
        LinearColor(ret)
    }
}
