//! The cylinder shape.

use crate::queries::{WgProjection, WgRay};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgRay, WgProjection),
    src = "cylinder.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader defining the cylinder shape as well as its ray-casting and point-projection functions.
pub struct WgCylinder;

#[cfg(test)]
mod test {
    use super::WgCylinder;
    use parry::shape::Cylinder;
    use wgcore::tensor::GpuVector;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_cylinder() {
        crate::queries::test_utils::test_point_projection::<WgCylinder, _>(
            "Cylinder",
            Cylinder::new(1.0, 0.5),
            |device, shapes, usages| GpuVector::init(device, shapes, usages).into_inner(),
        )
        .await;
    }
}
