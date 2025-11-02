//! Cylinder shape (3D only).
//!
//! A cylinder is defined by a half-height and a radius. The cylinder is centered at
//! the origin with its axis aligned along the Y-axis.
//!
//! **Note:** This shape is only available with the `dim3` feature.

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
/// GPU shader for the cylinder shape (3D only).
///
/// This shader provides WGSL implementations for:
/// - Ray-casting against cylinders.
/// - Point projection onto cylinder surfaces.
///
/// A cylinder is parameterized by:
/// - Half-height: Distance from the center to each flat circular end.
/// - Radius: Radius of the circular cross-section.
///
/// The cylinder is aligned along the Y-axis, extending from `-half_height` to
/// `+half_height`, with circular caps at both ends.
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
