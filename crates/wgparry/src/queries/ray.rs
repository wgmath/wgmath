//! Ray data structure and shader for ray-casting queries.
use wgcore::Shader;

#[derive(Shader)]
#[shader(src = "ray.wgsl")]
/// GPU shader defining the ray data structure for ray-casting queries.
///
/// This shader provides the WGSL definition of a ray, which consists of:
/// - **Origin**: The starting point of the ray.
/// - **Direction**: The direction vector (typically normalized).
pub struct WgRay;

wgcore::test_shader_compilation!(WgRay);
