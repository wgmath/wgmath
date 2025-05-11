use crate::math::Point;
use crate::substitute_aliases;
use wgcore::Shader;

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// The result of point-projection with a layout compatible with the corresponding wgsl struct.
pub struct GpuProjectionResult {
    /// The projected point.
    pub point: Point<f32>,
    /// Was the point to project inside the shape? (0 = false, other values = true)
    pub is_inside: u32,
    #[cfg(feature = "dim2")]
    /// Extra padding. Its value is irrelevant.
    pub padding: u32,
}

#[derive(Shader)]
#[shader(src = "projection.wgsl", src_fn = "substitute_aliases")]
/// Shader defining projection-related types.
pub struct WgProjection;

wgcore::test_shader_compilation!(WgProjection);

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::projection::GpuProjectionResult;
    use crate::{dim_shader_defs, substitute_aliases};
    use nalgebra::vector;
    #[cfg(feature = "dim2")]
    use parry2d::shape::Shape;
    #[cfg(feature = "dim3")]
    use parry3d::shape::Shape;
    use wgcore::kernel::{CommandEncoderExt, KernelDispatch};
    use wgcore::tensor::GpuVector;
    use wgcore::{gpu::GpuInstance, Shader};
    use wgpu::{Buffer, BufferUsages, Device};

    fn test_pipeline<Sh: Shader>(
        device: &Device,
        shader_shape_type: &str,
    ) -> wgpu::ComputePipeline {
        let test_kernel = substitute_aliases(&format!(
            r#"
struct ProjectionResultHostShareable {{
    point: Vector,
    is_inside: u32,
}}

@group(0) @binding(0)
var<storage, read> test_shapes: array<{shader_shape_type}>;
@group(0) @binding(1)
var<storage, read> test_points: array<Vector>;
@group(0) @binding(2)
var<storage, read_write> projs: array<Vector>;
@group(0) @binding(3)
var<storage, read_write> projs_on_boundary: array<ProjectionResultHostShareable>;

@compute @workgroup_size(1, 1, 1)
fn test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {{
    let i = invocation_id.x;
    let point = test_points[i];
    projs[i] = projectLocalPoint(test_shapes[i], point);

    let proj = projectLocalPointOnBoundary(test_shapes[i], point);
    projs_on_boundary[i] = ProjectionResultHostShareable(proj.point, u32(proj.is_inside));
}}
        "#
        ));

        let src = format!("{}\n{}", Sh::src(), test_kernel);
        let module = Sh::composer()
            .unwrap()
            .make_naga_module(naga_oil::compose::NagaModuleDescriptor {
                source: &src,
                file_path: Sh::FILE_PATH,
                shader_defs: dim_shader_defs(),
                ..Default::default()
            })
            .unwrap();
        wgcore::utils::load_module(device, "test", module)
    }

    pub async fn test_point_projection<Sh: Shader, S: Shape + Copy>(
        shader_shape_type: &str,
        shape: S,
        shape_buffer: impl FnOnce(&Device, &[S], BufferUsages) -> Buffer,
    ) {
        let gpu = GpuInstance::new().await.unwrap();
        let wg_ball = test_pipeline::<Sh>(gpu.device(), shader_shape_type);
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: u32 = 30;

        let mut points = vec![];
        let aabb = shape.compute_local_aabb();
        let step = aabb.half_extents() * 4.0 / (LEN as f32);
        let nk = if cfg!(feature = "dim2") { 1 } else { LEN };

        for i in 0..LEN {
            for j in 0..LEN {
                for _k in 0..nk {
                    let origin = aabb.mins.coords * 2.0;
                    #[cfg(feature = "dim2")]
                    let pt = vector![i as f32, j as f32].component_mul(&step) + origin;
                    #[cfg(feature = "dim3")]
                    let pt = (vector![i as f32, j as f32, _k as f32].component_mul(&step) + origin)
                        .push(0.0);
                    points.push(pt);
                }
            }
        }

        #[cfg(feature = "dim2")]
        type GpuPoint = na::Point2<f32>;
        #[cfg(feature = "dim3")]
        type GpuPoint = na::Point4<f32>;

        let usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let shapes = vec![shape; (LEN * LEN * LEN) as usize];
        let gpu_shapes = shape_buffer(gpu.device(), &shapes, usages);
        let gpu_points = GpuVector::init(gpu.device(), &points, usages);
        let gpu_projs: GpuVector<GpuPoint> =
            GpuVector::uninit(gpu.device(), points.len() as u32, usages);
        let gpu_projs_on_boundary: GpuVector<GpuProjectionResult> =
            GpuVector::uninit(gpu.device(), points.len() as u32, usages);

        let usages = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
        let staging_projs: GpuVector<GpuPoint> =
            GpuVector::uninit(gpu.device(), points.len() as u32, usages);
        let staging_projs_on_boundary: GpuVector<GpuProjectionResult> =
            GpuVector::uninit(gpu.device(), points.len() as u32, usages);

        let mut pass = encoder.compute_pass("test", None);
        KernelDispatch::new(gpu.device(), &mut pass, &wg_ball)
            .bind0([
                &gpu_shapes,
                gpu_points.buffer(),
                gpu_projs.buffer(),
                gpu_projs_on_boundary.buffer(),
            ])
            .dispatch(points.len() as u32);
        drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

        staging_projs.copy_from(&mut encoder, &gpu_projs);
        staging_projs_on_boundary.copy_from(&mut encoder, &gpu_projs_on_boundary);
        gpu.queue().submit(Some(encoder.finish()));

        let result_projs = staging_projs.read(gpu.device()).await.unwrap();
        let gpu_result_projs_on_boundary =
            staging_projs_on_boundary.read(gpu.device()).await.unwrap();

        #[cfg(feature = "dim2")]
        for (i, pt) in points.iter().enumerate() {
            let proj = shape.project_local_point(&(*pt).into(), true);
            approx::assert_relative_eq!(proj.point, result_projs[i], epsilon = 1.0e-6);

            let proj = shape.project_local_point(&(*pt).into(), false);
            if !proj.point.x.is_finite() {
                continue;
            }

            assert_eq!(
                proj.is_inside,
                gpu_result_projs_on_boundary[i].is_inside != 0
            );
            approx::assert_relative_eq!(
                proj.point,
                gpu_result_projs_on_boundary[i].point,
                epsilon = 1.0e-6
            );
        }

        #[cfg(feature = "dim3")]
        for (i, pt) in points.iter().enumerate() {
            let proj = shape.project_local_point(&pt.xyz().into(), true);
            approx::assert_relative_eq!(proj.point, result_projs[i].xyz(), epsilon = 1.0e-6);

            let proj = shape.project_local_point(&pt.xyz().into(), false);
            if !proj.point.x.is_finite() {
                continue;
            }

            assert_eq!(
                proj.is_inside,
                gpu_result_projs_on_boundary[i].is_inside != 0
            );
            approx::assert_relative_eq!(
                proj.point,
                gpu_result_projs_on_boundary[i].point,
                epsilon = 1.0e-6
            );
        }
    }
}
