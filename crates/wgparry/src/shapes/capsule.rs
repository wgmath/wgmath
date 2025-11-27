//! Capsule shape - swept sphere along a line segment.
//!
//! A capsule is defined by a line segment (from point A to point B) and a radius.
//! It can be visualized as a sphere swept along the segment, or as a cylinder with
//! hemispherical caps at both ends.

use crate::queries::{WgPolygonalFeature, WgProjection, WgRay};
use crate::shapes::WgSegment;
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgRay, WgProjection, WgSegment, WgPolygonalFeature),
    src = "capsule.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// GPU shader for the capsule shape.
///
/// This shader provides WGSL implementations for:
/// - Ray-casting against capsules.
/// - Point projection onto capsule surfaces.
///
/// A capsule is parameterized by:
/// - A line segment (two endpoints A and B).
/// - A radius.
///
/// The capsule surface is the set of all points at distance `radius` from the segment.
pub struct WgCapsule;

#[cfg(test)]
mod test {
    use super::WgCapsule;
    use parry::shape::Capsule;
    use wgcore::tensor::GpuVector;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_capsule() {
        crate::queries::test_utils::test_point_projection::<WgCapsule, _>(
            "Capsule",
            Capsule::new_y(1.0, 0.5),
            |device, shapes, usages| GpuVector::encase(device, shapes, usages).into_inner(),
        )
        .await;
    }
}
