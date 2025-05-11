//! The triangle shape.

use crate::substitute_aliases;
use wgcore::Shader;

#[derive(Shader)]
#[shader(src = "triangle.wgsl", src_fn = "substitute_aliases")]
/// Shader defining the triangle shape as well as its ray-casting and point-projection functions.
pub struct WgTriangle;
