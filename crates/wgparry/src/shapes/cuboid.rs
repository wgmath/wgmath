//! The cuboid shape.

use crate::queries::{WgProjection, WgRay};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgRay, WgProjection),
    src = "cuboid.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader defining the Cuboid shape as well as its ray-casting and point-projection functions.
pub struct WgCuboid;

#[cfg(test)]
mod test {
    use super::WgCuboid;
    use na::vector;
    #[cfg(feature = "dim2")]
    use parry2d::shape::Cuboid;
    #[cfg(feature = "dim3")]
    use parry3d::shape::Cuboid;
    use wgcore::tensor::GpuVector;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_cuboid() {
        crate::queries::test_utils::test_point_projection::<WgCuboid, _>(
            "Cuboid",
            #[cfg(feature = "dim2")]
            Cuboid::new(vector![1.0, 2.0]),
            #[cfg(feature = "dim3")]
            Cuboid::new(vector![1.0, 2.0, 3.0]),
            |device, shapes, usages| GpuVector::encase(device, shapes, usages).into_inner(),
        )
        .await;
    }
}
