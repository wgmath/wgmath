//! The ray structure.

use wgcore::Shader;

#[derive(Shader)]
#[shader(src = "ray.wgsl")]
/// Shader defining the wgsl ray structure for ray-casting.
pub struct WgRay;

wgcore::test_shader_compilation!(WgRay);
