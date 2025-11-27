//! Cone shape (3D only).
//!
//! A cone is defined by a half-height (distance from base to apex) and a base radius.
//! The cone is centered at the origin with its axis aligned along the Y-axis, with the
//! apex pointing in the positive Y direction.
//!
//! **Note:** This shape is only available with the `dim3` feature.

use crate::queries::{WgPolygonalFeature, WgProjection, WgRay};
use crate::shapes::segment::WgSegment;
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgRay, WgProjection, WgSegment, WgPolygonalFeature),
    src = "cone.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// GPU shader for the cone shape (3D only).
///
/// This shader provides WGSL implementations for:
/// - Ray-casting against cones
/// - Point projection onto cone surfaces
///
/// A cone is parameterized by:
/// - Half-height: Distance from the center to the apex (and to the base)
/// - Radius: Radius of the circular base
///
/// The cone is aligned along the Y-axis with the apex at `+half_height` and
/// the base centered at `-half_height`.
pub struct WgCone;

#[cfg(test)]
mod test {
    use super::WgCone;
    use parry::shape::Cone;
    use wgcore::tensor::GpuVector;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_cone() {
        crate::queries::test_utils::test_point_projection::<WgCone, _>(
            "Cone",
            Cone::new(1.0, 0.5),
            |device, shapes, usages| GpuVector::init(device, shapes, usages).into_inner(),
        )
        .await;
    }
}
