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
