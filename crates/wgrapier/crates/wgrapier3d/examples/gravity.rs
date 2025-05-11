use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use naga_oil::compose::NagaModuleDescriptor;
use nalgebra::{Similarity3, Vector3, Vector4};
use wgcore::gpu::GpuInstance;
use wgcore::kernel::{CommandEncoderExt, KernelDispatch};
use wgcore::tensor::GpuVector;
use wgcore::utils::{load_module, load_shader};
use wgcore::Shader;
use wgpu::{BufferUsages, CommandEncoder, ComputePipeline};
use wgrapier3d::dynamics::body::{GpuMassProperties, GpuVelocity, WgBody};

#[derive(Resource)]
struct Gpu {
    instance: GpuInstance,
}

#[derive(Resource)]
struct PhysicsContext {
    // GPU vectors.
    wg_poses: GpuVector<Similarity3<f32>>,
    wg_local_mprops: GpuVector<GpuMassProperties>,
    wg_mprops: GpuVector<GpuMassProperties>,
    wg_vels: GpuVector<GpuVelocity>,
    wg_readback_poses: GpuVector<Similarity3<f32>>,
    rb_poses: Vec<Similarity3<f32>>,
    simulation: ComputePipeline,
}

#[derive(Component)]
struct RigidBodyId(pub usize);

#[async_std::main]
pub async fn main() {
    let instance = GpuInstance::new().await.unwrap();

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .insert_resource(Gpu { instance })
        .add_systems(Startup, (setup_physics, setup_graphics).chain())
        .add_systems(Update, step_simulation)
        .run();
}

fn step_simulation(
    gpu: Res<Gpu>,
    physics: Res<PhysicsContext>,
    mut bodies: Query<(&mut Transform, &RigidBodyId)>,
) {
    let t0 = std::time::Instant::now();
    let gpu = &gpu.instance;
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    let mut pass = encoder.compute_pass("step_simulation", None);
    KernelDispatch::new(gpu.device(), &mut pass, &physics.simulation)
        .bind0([
            physics.wg_mprops.buffer(),
            physics.wg_local_mprops.buffer(),
            physics.wg_poses.buffer(),
            physics.wg_vels.buffer(),
        ])
        .dispatch(256);
    drop(pass);
    gpu.queue().submit(Some(encoder.finish()));
    gpu.device().poll(wgpu::Maintain::Wait);
    println!("Simulation time: {}.", t0.elapsed().as_secs_f32() * 1000.0);

    // Very unoptimized readback.

    let t0 = std::time::Instant::now();
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    physics
        .wg_readback_poses
        .copy_from(&mut encoder, &physics.wg_poses);
    gpu.queue().submit(Some(encoder.finish()));
    let new_poses =
        async_std::task::block_on(physics.wg_readback_poses.read(gpu.device())).unwrap();

    println!(
        "Read back {} poses ({}ms).",
        new_poses.len(),
        t0.elapsed().as_secs_f32() * 1000.0
    );

    for (mut body, id) in bodies.iter_mut() {
        let pos = new_poses[id.0].isometry.translation;
        body.translation = Vec3::new(pos.x, pos.y, pos.z);
    }
}

fn setup_physics(mut commands: Commands, gpu: Res<Gpu>) {
    const NXZ: usize = 100;
    const NY: usize = 10;
    const NUM_BODIES: usize = NXZ * NXZ * NY;

    let gpu = &gpu.instance;
    let mut rb_poses = vec![Similarity3::identity(); NUM_BODIES];
    let mut rb_local_mprops = vec![GpuMassProperties::default(); NUM_BODIES];
    let mut rb_mprops = vec![GpuMassProperties::default(); NUM_BODIES];
    let rb_vels = vec![GpuVelocity::default(); NUM_BODIES];

    for i in 0..NXZ {
        for j in 0..NY {
            for k in 0..NXZ {
                let elt = i + k * NXZ + j * NXZ * NXZ;
                let pos = Vector3::new(i as f32, j as f32, k as f32);
                rb_poses[elt].isometry.translation.vector = pos.xyz();
                rb_mprops[elt].com = pos;
            }
        }
    }

    let ctxt = PhysicsContext {
        wg_vels: GpuVector::encase(gpu.device(), &rb_vels, BufferUsages::STORAGE),
        wg_local_mprops: GpuVector::encase(gpu.device(), &rb_local_mprops, BufferUsages::STORAGE),
        wg_mprops: GpuVector::encase(gpu.device(), &rb_mprops, BufferUsages::STORAGE),
        wg_poses: GpuVector::init(
            gpu.device(),
            &rb_poses,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ),
        wg_readback_poses: GpuVector::uninit(
            gpu.device(),
            rb_poses.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        ),
        rb_poses,
        simulation: load_simulation_shader(gpu),
    };
    commands.insert_resource(ctxt);
}

#[derive(Shader)]
#[shader(derive(WgBody), src = "./gravity.wgsl", composable = false)]
struct WgGravity {
    main: ComputePipeline,
}

fn load_simulation_shader(gpu: &GpuInstance) -> ComputePipeline {
    WgGravity::from_device(gpu.device()).unwrap().main
}

/// set up a simple 3D scene
fn setup_graphics(
    mut commands: Commands,
    physics: Res<PhysicsContext>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let colors = [
        Color::srgb_u8(124, 144, 255),
        Color::srgb_u8(8, 144, 255),
        Color::srgb_u8(124, 7, 255),
        Color::srgb_u8(124, 144, 7),
        Color::srgb_u8(200, 37, 255),
        Color::srgb_u8(124, 230, 25),
    ];
    let materials = colors.map(|c| materials.add(c));
    let sphere = meshes.add(Sphere::new(0.5));

    for (rb_id, pose) in physics.rb_poses.iter().enumerate() {
        commands.spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(materials[rb_id % colors.len()].clone()),
            Transform::from_xyz(
                pose.isometry.translation.x,
                pose.isometry.translation.y,
                pose.isometry.translation.z,
            ),
            RigidBodyId(rb_id),
        ));
    }

    // light
    commands.insert_resource(AmbientLight {
        brightness: 1000.0,
        ..Default::default()
    });

    // camera
    commands.spawn((
        Camera3d::default(),
        Msaa::Sample4,
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        PanOrbitCamera::default(),
    ));
}
