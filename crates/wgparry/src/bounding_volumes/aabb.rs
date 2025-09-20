use crate::shapes::WgShape;
use crate::substitute_aliases;
use wgcore::{test_shader_compilation, Shader};
use wgebra::WgSim3;

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgShape),
    src = "./aabb.wgsl",
    src_fn = "substitute_aliases"
)]
pub struct WgAabb;

test_shader_compilation!(WgAabb);
