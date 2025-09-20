use bevy::ecs::system::RegisteredSystemError::SelfRemove;
use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bytemuck::Pod;
use encase::internal::{CreateFrom, ReadFrom, WriteInto};
use encase::{ShaderSize, ShaderType};
use naga_oil::compose::NagaModuleDescriptor;
use nalgebra::{Similarity3, Vector3, Vector4};
use rapier3d::geometry::Aabb;
use wgcore::gpu::GpuInstance;
use wgcore::indirect::DispatchIndirectArgs;
use wgcore::kernel::{CommandEncoderExt, KernelDispatch};
use wgcore::tensor::{GpuScalar, GpuTensor, GpuVector, TensorBuilder};
use wgcore::utils::{load_module, load_shader};
use wgcore::Shader;
use wgpu::{Buffer, BufferUsages, CommandEncoder, ComputePipeline, Device};
use wgrapier3d::dynamics::body::{
    GpuLocalMassProperties, GpuVelocity, GpuWorldMassProperties, WgBody,
};
use wgrapier3d::dynamics::{
    prefix_sum::{PrefixSumWorkspace, WgPrefixSum},
    GpuSimParams, GpuTwoBodyConstraint, JacobiSolverArgs, WgSimParams, WgSolverJacobi,
};
use wgrapier3d::wgparry::{
    broad_phase::{Lbvh, WgBruteForceBroadPhase, WgLbvh, WgNarrowPhase},
    queries::GpuIndexedContact,
    shapes::GpuShape,
};

#[derive(Resource)]
struct Gpu {
    instance: GpuInstance,
}

#[derive(Resource)]
struct PhysicsContext {
    // GPU vectors.
    wg_sim_params: GpuScalar<GpuSimParams>,
    wg_poses: GpuVector<Similarity3<f32>>,
    wg_local_mprops: GpuVector<GpuLocalMassProperties>,
    wg_mprops: GpuVector<GpuWorldMassProperties>,
    wg_vels: GpuVector<GpuVelocity>,
    wg_solver_vels: GpuVector<GpuVelocity>,
    wg_solver_vels_out: GpuVector<GpuVelocity>,
    wg_readback_poses: GpuVector<Similarity3<f32>>,

    wg_shapes: GpuVector<GpuShape>,
    wg_num_shapes: GpuScalar<u32>,
    wg_num_shapes_indirect: GpuScalar<[u32; 3]>,
    wg_collision_pairs: GpuVector<[u32; 2]>,
    wg_collision_pairs_len: GpuScalar<u32>,
    wg_collision_pairs_indirect: GpuScalar<DispatchIndirectArgs>,
    wg_contacts: GpuVector<GpuIndexedContact>,
    wg_contacts_len: GpuScalar<u32>,
    wg_contacts_indirect: GpuScalar<DispatchIndirectArgs>,
    wg_constraints: GpuVector<GpuTwoBodyConstraint>,
    wg_constraints_counts: GpuVector<u32>,
    wg_body_constraint_ids: GpuVector<u32>,
    prefix_sum_workspace: PrefixSumWorkspace,

    debug_aabb_mins: GpuVector<Vector4<f32>>,
    debug_aabb_maxs: GpuVector<Vector4<f32>>,

    rb_poses: Vec<Similarity3<f32>>,
    gravity: WgGravity,
    broad_phase: WgBruteForceBroadPhase,
    narrow_phase: WgNarrowPhase,
    solver: WgSolverJacobi,
    prefix_sum: WgPrefixSum,
    lbvh: Lbvh,
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
    mut physics: ResMut<PhysicsContext>,
    mut bodies: Query<(&mut Transform, &RigidBodyId)>,
) {
    let physics = &mut *physics;
    let t0 = std::time::Instant::now();
    let gpu = &gpu.instance;
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    let mut pass = encoder.compute_pass("step_simulation", None);
    KernelDispatch::new(gpu.device(), &mut pass, &physics.gravity.main)
        .bind0([
            physics.wg_mprops.buffer(),
            physics.wg_local_mprops.buffer(),
            physics.wg_poses.buffer(),
            physics.wg_vels.buffer(),
        ])
        .bind(1, [physics.wg_sim_params.buffer()])
        .dispatch(256);

    // physics.broad_phase.dispatch(
    //     gpu.device(),
    //     &mut pass,
    //     physics.wg_poses.len() as u32,
    //     &physics.wg_poses,
    //     &physics.wg_shapes,
    //     &physics.wg_num_shapes,
    //     &physics.wg_collision_pairs,
    //     &physics.wg_collision_pairs_len,
    //     &physics.wg_collision_pairs_indirect,
    //     &physics.debug_aabb_mins,
    //     &physics.debug_aabb_maxs,
    // );

    physics.lbvh.launch(
        gpu.device(),
        &mut pass,
        physics.wg_poses.len() as u32,
        &physics.wg_poses,
        &physics.wg_shapes,
        &physics.wg_num_shapes,
        &physics.wg_collision_pairs,
        &physics.wg_collision_pairs_len,
        &physics.wg_collision_pairs_indirect,
    );

    physics.narrow_phase.dispatch(
        gpu.device(),
        &mut pass,
        physics.wg_poses.len() as u32,
        &physics.wg_poses,
        &physics.wg_shapes,
        &physics.wg_collision_pairs,
        &physics.wg_collision_pairs_len,
        &physics.wg_collision_pairs_indirect,
        &physics.wg_contacts,
        &physics.wg_contacts_len,
        &physics.wg_contacts_indirect,
    );

    let jacobi_args = JacobiSolverArgs {
        num_colliders: physics.wg_poses.len() as u32,
        contacts: &physics.wg_contacts,
        contacts_len: &physics.wg_contacts_len,
        contacts_len_indirect: &physics.wg_contacts_indirect,
        constraints: &physics.wg_constraints,
        sim_params: &physics.wg_sim_params,
        colliders_len: &physics.wg_num_shapes,
        colliders_len_indirect: &physics.wg_num_shapes_indirect,
        poses: &physics.wg_poses,
        vels: &physics.wg_vels,
        solver_vels: &physics.wg_solver_vels,
        solver_vels_out: &physics.wg_solver_vels_out,
        mprops: &physics.wg_mprops,
        body_constraint_counts: &physics.wg_constraints_counts,
        body_constraint_ids: &physics.wg_body_constraint_ids,
        prefix_sum: &physics.prefix_sum,
        prefix_sum_workspace: &mut physics.prefix_sum_workspace,
    };

    physics
        .solver
        .dispatch(gpu.device(), &mut pass, jacobi_args);

    drop(pass);
    gpu.queue().submit(Some(encoder.finish()));
    gpu.device().poll(wgpu::Maintain::Wait);
    println!("Simulation time: {}.", t0.elapsed().as_secs_f32() * 1000.0);

    // TODO: very unoptimized readback.

    let t0 = std::time::Instant::now();
    let new_poses = async_std::task::block_on(physics.wg_poses.slow_read(&gpu));

    println!(
        "Read back {} poses ({}ms).",
        new_poses.len(),
        t0.elapsed().as_secs_f32() * 1000.0
    );

    // let constraints_counts = read_buffer(&gpu, &physics.wg_constraints_counts);
    // println!("Constraints counts: {:?}", &constraints_counts);

    // let collision_pairs_len = read_buffer(&gpu, &physics.wg_collision_pairs_len);
    // let constraints_len = read_buffer(&gpu, &physics.wg_contacts_len);
    // let debug_mins = read_buffer(&gpu, &physics.debug_aabb_mins);
    // let debug_maxs = read_buffer(&gpu, &physics.debug_aabb_maxs);
    // let constraints = read_buffer_encase(&gpu, &physics.wg_constraints);
    // let contacts = read_buffer_encase(&gpu, &physics.wg_contacts);
    // println!(
    //     "Coll pair len: {}, constraint len: {}",
    //     collision_pairs_len[0], constraints_len[0]
    // );

    // if constraints[0].dir_a.y != 0.0 {
    //     println!("constraints: {:?}", &contacts[..2]);
    //     panic!("");
    // }

    // let reconstructed_aabbs: Vec<_> = debug_mins
    //     .iter()
    //     .zip(debug_maxs.iter())
    //     .map(|(mins, maxs)| Aabb::new(mins.xyz().into(), maxs.xyz().into()))
    //     .collect();
    // let mut ninters = 0;
    // for i in 0..reconstructed_aabbs.len() {
    //     for j in 0..reconstructed_aabbs.len() {
    //         use rapier3d::geometry::BoundingVolume;
    //         if i != j && reconstructed_aabbs[i].intersects(&reconstructed_aabbs[j]) {
    //             ninters += 1;
    //         }
    //     }
    // }
    // println!("Inters: {}", ninters);
    // println!("Aabb mins: {:?}, maxs: {:?}", debug_mins, debug_maxs);

    for (mut body, id) in bodies.iter_mut() {
        let pos = new_poses[id.0].isometry.translation;
        body.translation = Vec3::new(pos.x, pos.y, pos.z);
    }
}

fn setup_physics(mut commands: Commands, gpu: Res<Gpu>) {
    const NXZ: isize = 100;
    const NY: isize = 10;

    let gpu = &gpu.instance;
    let mut rb_poses = Vec::new();
    let mut rb_local_mprops = Vec::new();
    let mut rb_mprops = Vec::new();

    let mut id = 0;
    for j in 0..NY {
        let max_ik = NXZ / 2;
        for i in -max_ik..max_ik {
            for k in -max_ik..max_ik {
                let is_static =
                    i == -max_ik || i == max_ik - 1 || k == -max_ik || k == max_ik - 1 || j == 0;
                let y = if is_static { j as f32 } else { j as f32 * 2.0 };
                let pos = Vector3::new(i as f32, y, k as f32);
                let mut pose = Similarity3::identity();
                let mut local_mprops = GpuLocalMassProperties::default();
                let mut mprops = GpuWorldMassProperties::default();
                pose.isometry.translation.vector = pos.xyz();
                mprops.com = pos;

                if is_static {
                    local_mprops.inv_mass.fill(0.0);
                    local_mprops.inv_principal_inertia_sqrt.fill(0.0);
                    mprops.inv_mass.fill(0.0);
                    mprops.inv_inertia_sqrt.fill(0.0);
                }

                rb_poses.push(pose);
                rb_local_mprops.push(local_mprops);
                rb_mprops.push(mprops);

                id += 1;
            }
        }
    }

    let num_bodies = rb_poses.len();
    let rb_vels = vec![GpuVelocity::default(); num_bodies];
    let shapes = vec![GpuShape::ball(0.5); num_bodies];
    let STORAGE: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let wg_shapes = GpuVector::init(gpu.device(), &shapes, STORAGE);
    let wg_num_shapes = GpuScalar::init(gpu.device(), rb_poses.len() as u32, BufferUsages::UNIFORM);
    let wg_num_shapes_indirect = GpuScalar::init(
        gpu.device(),
        [rb_poses.len().div_ceil(64) as u32, 1, 1],
        BufferUsages::STORAGE | BufferUsages::INDIRECT,
    );

    let estimated_contact_count = (rb_poses.len() * 30) as u32;

    let wg_collision_pairs = GpuVector::uninit(gpu.device(), estimated_contact_count, STORAGE);
    let wg_collision_pairs_len = GpuScalar::uninit(gpu.device(), STORAGE);
    let wg_collision_pairs_indirect =
        GpuScalar::uninit(gpu.device(), STORAGE | BufferUsages::INDIRECT);

    let wg_contacts = GpuVector::uninit_encased(gpu.device(), estimated_contact_count, STORAGE);
    let wg_contacts_len = GpuScalar::uninit(gpu.device(), STORAGE);
    let wg_contacts_indirect = GpuScalar::uninit(gpu.device(), STORAGE | BufferUsages::INDIRECT);
    let wg_constraints = GpuVector::uninit_encased(gpu.device(), estimated_contact_count, STORAGE);
    let wg_constraints_counts =
        GpuVector::uninit_encased(gpu.device(), rb_poses.len() as u32, STORAGE);
    let wg_body_constraint_ids =
        GpuVector::uninit_encased(gpu.device(), estimated_contact_count * 2, STORAGE);
    let sim_params = GpuSimParams::pgs_legacy();

    let ctxt = PhysicsContext {
        wg_sim_params: GpuScalar::init(gpu.device(), sim_params, STORAGE | BufferUsages::UNIFORM),
        wg_vels: GpuVector::encase(gpu.device(), &rb_vels, STORAGE),
        wg_solver_vels: GpuVector::encase(gpu.device(), &rb_vels, STORAGE),
        wg_solver_vels_out: GpuVector::encase(gpu.device(), &rb_vels, STORAGE),
        wg_local_mprops: GpuVector::encase(gpu.device(), &rb_local_mprops, STORAGE),
        wg_mprops: GpuVector::encase(gpu.device(), &rb_mprops, STORAGE),
        wg_poses: GpuVector::init(gpu.device(), &rb_poses, STORAGE | BufferUsages::COPY_SRC),
        wg_readback_poses: GpuVector::uninit(
            gpu.device(),
            rb_poses.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        ),
        wg_shapes,
        wg_num_shapes,
        wg_num_shapes_indirect,
        wg_collision_pairs,
        wg_collision_pairs_len,
        wg_collision_pairs_indirect,
        wg_contacts,
        wg_contacts_len,
        wg_contacts_indirect,
        wg_constraints,
        wg_constraints_counts,
        wg_body_constraint_ids,
        prefix_sum_workspace: PrefixSumWorkspace::default(),
        debug_aabb_mins: GpuVector::uninit(gpu.device(), rb_poses.len() as u32, STORAGE),
        debug_aabb_maxs: GpuVector::uninit(gpu.device(), rb_poses.len() as u32, STORAGE),
        rb_poses,
        gravity: WgGravity::from_device(gpu.device()).unwrap(),
        broad_phase: WgBruteForceBroadPhase::from_device(gpu.device()).unwrap(),
        narrow_phase: WgNarrowPhase::from_device(gpu.device()).unwrap(),
        solver: WgSolverJacobi::from_device(gpu.device()).unwrap(),
        prefix_sum: WgPrefixSum::from_device(gpu.device()).unwrap(),
        lbvh: Lbvh::new(gpu.device()).unwrap(),
    };
    commands.insert_resource(ctxt);
}

#[derive(Shader)]
#[shader(
    derive(WgBody, WgSimParams),
    src = "./gravity.wgsl",
    composable = false
)]
struct WgGravity {
    main: ComputePipeline,
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
