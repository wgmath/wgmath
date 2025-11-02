//! Ball shape - sphere (3D) or circle (2D).
//!
//! The ball is the simplest geometric primitive, defined by a single radius parameter.
//! In 3D, this represents a sphere; in 2D, a circle.

use crate::queries::{WgProjection, WgRay};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgRay, WgProjection),
    src = "ball.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// GPU shader for the ball (sphere/circle) shape.
///
/// This shader provides WGSL implementations for:
/// - Ray-casting against balls
/// - Point projection onto ball surfaces
///
/// The ball is defined by a single radius parameter and is centered at the origin
/// in local space. Use transformations to position and scale balls in world space.
pub struct WgBall;

#[cfg(test)]
mod test {
    use super::WgBall;
    #[cfg(feature = "dim2")]
    use parry2d::shape::Ball;
    #[cfg(feature = "dim3")]
    use parry3d::shape::Ball;
    use wgcore::tensor::GpuVector;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_ball() {
        crate::queries::test_utils::test_point_projection::<WgBall, _>(
            "Ball",
            Ball::new(0.5),
            |device, shapes, usages| GpuVector::init(device, shapes, usages).into_inner(),
        )
        .await;
    }
}
