//! The ray structure.

use crate::math::{Point, Vector};
use crate::shapes::WgBall;
use encase::ShaderType;
use na::Vector2;
use wgcore::Shader;
use wgebra::WgSim3;

#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuContact {
    point_a: Vector<f32>,
    point_b: Vector<f32>,
    normal_a: Vector<f32>,
    normal_b: Vector<f32>,
    dist: f32,
}

#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuIndexedContact {
    contact: GpuContact,
    colliders: Vector2<u32>,
}

#[derive(Shader)]
#[shader(derive(WgBall, WgSim3), src = "contact.wgsl")]
/// Shader implementing contact definition and contact calculation.
pub struct WgContact;

wgcore::test_shader_compilation!(WgContact);
