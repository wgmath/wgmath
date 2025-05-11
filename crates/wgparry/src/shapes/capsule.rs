//! The capsule shape.

use crate::queries::{WgProjection, WgRay};
use crate::shapes::WgSegment;
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgRay, WgProjection, WgSegment),
    src = "capsule.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader defining the capsule shape as well as its ray-casting and point-projection functions.
pub struct WgCapsule;

#[cfg(test)]
mod test {
    use super::WgCapsule;
    use parry::shape::Capsule;
    use wgcore::tensor::GpuVector;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_capsule() {
        crate::projection::test_utils::test_point_projection::<WgCapsule, _>(
            "Capsule",
            Capsule::new_y(1.0, 0.5),
            |device, shapes, usages| GpuVector::encase(device, shapes, usages).into_inner(),
        )
        .await;
    }
}
