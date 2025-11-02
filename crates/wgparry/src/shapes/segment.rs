//! Line segment shape.
//!
//! A segment is a straight line connecting two points A and B. It's a fundamental
//! building block used by other shapes like capsules and is useful for ray-casting
//! and distance queries.

use crate::queries::{WgProjection, WgRay};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgRay, WgProjection),
    src = "segment.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// GPU shader for the line segment shape.
///
/// This shader provides WGSL implementations for:
/// - Ray-casting against segments.
/// - Point projection onto segments (finding the closest point on the segment).
///
/// A segment is defined by two endpoints A and B. Point projection operations
/// find the closest point on the segment to a given query point, which may be
/// one of the endpoints or an interior point.
pub struct WgSegment;

// TODO:
// #[cfg(test)]
// mod test {
//     use super::WgSegment;
//     use parry::shape::Segment;
//     use wgcore::tensor::GpuVector;
//
//     #[futures_test::test]
//     #[serial_test::serial]
//     async fn gpu_segment() {
//         crate::projection::test_utils::test_point_projection::<WgSegment, _>(
//             "Segment",
//             Segment::new(1.0, 0.5),
//             |device, shapes, usages| GpuVector::encase(device, shapes, usages).into_inner(),
//         )
//         .await;
//     }
// }
