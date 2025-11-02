//! Mass properties update kernel (currently used for mprops sync)
//!
//! This shader updates world-space mass properties for all rigid bodies based on
//! their current poses and local mass properties.


#import wgrapier::body as Body;
#import wgrapier::dynamics::sim_params as Params;

#if DIM == 2
    #import wgebra::sim2 as Pose;
#else
    #import wgebra::sim3 as Pose;
#endif


/// World-space mass properties (output).
/// Updated by transforming local mass properties to world space.
@group(0) @binding(0)
var<storage, read_write> mprops: array<Body::WorldMassProperties>;

/// Local-space mass properties (input, constant).
/// These are defined in the body's local coordinate frame.
@group(0) @binding(1)
var<storage, read> local_mprops: array<Body::LocalMassProperties>;

/// Rigid body poses (input).
/// Used to transform mass properties from local to world space.
#if DIM == 2
@group(0) @binding(2)
var<storage, read_write> poses: array<Pose::Sim2>;
#else
@group(0) @binding(2)
var<storage, read_write> poses: array<Pose::Sim3>;
#endif

/// Rigid body velocities (unused in this kernel).
@group(0) @binding(3)
var<storage, read_write> vels: array<Body::Velocity>;

/// Simulation parameters (unused in this kernel, but part of bind group).
@group(1) @binding(0)
var<uniform> params: Params::SimParams;

/// Workgroup size: 64 threads per workgroup.
const WORKGROUP_SIZE: u32 = 64;

/// Updates world-space mass properties for all rigid bodies.
///
/// For each body:
/// 1. Transform center of mass from local to world space
/// 2. Transform inertia tensor to world orientation (3D only)
///
/// In 2D: Only COM needs transformation (inertia is scalar)
/// In 3D: Both COM and inertia tensor need transformation
///
/// @param invocation_id: Global thread ID
/// @param num_workgroups: Number of workgroups dispatched
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // Total number of threads across all workgroups
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;

    // Grid-stride loop: each thread processes multiple bodies if necessary
    for (var i = invocation_id.x; i < arrayLength(&poses); i += num_threads) {
        // Transform mass properties from local to world space
        // - Transforms COM position by pose
        // - Rotates inertia tensor to world orientation (3D)
        let new_mprops = Body::updateMprops(poses[i], local_mprops[i]);

        // Write updated world-space mass properties
        mprops[i] = new_mprops;
    }
}
