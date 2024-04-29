use cgmath::{
    prelude::*,
    Matrix3,
    Matrix4,
};
use crate::utils::Transform;

type Mat4 = Matrix4<f32>;

pub trait TransformExt {
    fn to_matrix(&self) -> Mat4;
}

impl TransformExt for Transform {
    fn to_matrix(&self) -> Mat4 {
        match self {
            Transform::Normal => Mat4::one(),
            // Transform::Flipped => Mat4::from_nonuniform_scale(1.0, -1.0, 1.0),
            _ => todo!(),
        }
    }
}
