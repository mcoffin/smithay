use cgmath::{
    prelude::*,
    Matrix4,
};
use crate::utils::Transform;

type Mat4 = Matrix4<f32>;

pub trait TransformExt {
    fn to_matrix(&self) -> Option<Mat4>;
}

impl TransformExt for Transform {
    fn to_matrix(&self) -> Option<Mat4> {
        match self {
            Transform::Normal => None,
            Transform::Flipped => Some(Mat4::from_nonuniform_scale(1.0, -1.0, 1.0)),
            _ => todo!(),
        }
    }
}

/// Matrix that equates to 2D scale by 2.0, and translate by (-1.0, -1.0)
///
/// translates 0-based coordinates to 0-centered coordinates for rectangle ops
pub const MAT4_MODEL_BOX: Matrix4::<f32> = Matrix4::new(
    2.0, 0.0, 0.0, 0.0,
    0.0, 2.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    -1.0, -1.0, 0.0, 1.0
);
