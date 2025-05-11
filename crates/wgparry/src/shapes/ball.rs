//! The ball shape.

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
/// Shader defining the ball shape as well as its ray-casting and point-projection functions.
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
        crate::projection::test_utils::test_point_projection::<WgBall, _>(
            "Ball",
            Ball::new(0.5),
            |device, shapes, usages| GpuVector::init(device, shapes, usages).into_inner(),
        )
        .await;
    }
}
