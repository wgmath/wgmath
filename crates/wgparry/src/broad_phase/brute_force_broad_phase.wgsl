//! Brute Force Broad Phase Collision Detection
//!
//! This compute shader implements O(n²) all-pairs collision detection using AABBs.
//! While not scalable to large scenes, it's very convenient for testing and debugging.
//!
//! Pipeline:
//! 1. reset: Clears collision pair counter
//! 2. main: Double-nested loop tests all pairs (i < j) for AABB overlap
//! 3. init_indirect_args: Prepares indirect dispatch for narrow phase
//!
//! Buffers:
//! - Input: num_colliders (uniform), poses, shapes (storage)
//! - Output: collision_pairs (vec2<u32> array), collision_pairs_len (atomic counter)

#import wgparry::shape as Shape;
#import wgparry::bounding_volumes::aabb as Aabb;
#import wgcore::indirect as Indirect;

#if DIM == 2
    #import wgebra::sim2 as Pose;
#else
    #import wgebra::sim3 as Pose;
#endif

@group(0) @binding(0)
var<uniform> num_colliders: u32;
@group(0) @binding(1)
var<storage, read> poses: array<Transform>;
@group(0) @binding(2)
var<storage, read> shapes: array<Shape::Shape>;
@group(0) @binding(3)
var<storage, read_write> collision_pairs: array<vec2<u32>>;
@group(0) @binding(4)
var<storage, read_write> collision_pairs_len: atomic<u32>;
@group(0) @binding(5)
var<storage, read_write> collision_pairs_len_indirect_args: Indirect::DispatchIndirectArgs;

@group(1) @binding(0)
var<storage, read_write> debug_mins: array<Vector>;
@group(1) @binding(1)
var<storage, read_write> debug_maxs: array<Vector>;


const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(1, 1, 1)
fn reset() {
    collision_pairs_len = 0u;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn debug_compute_aabb(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;

    if i < arrayLength(&debug_mins) {
        let pose1 = poses[i];
        let shape1 = shapes[i];
        var aabb1 = Shape::aabb(pose1, shape1);
        debug_mins[i] = aabb1.mins;
        debug_maxs[i] = aabb1.maxs;
    }
}

@compute @workgroup_size(1, 1, 1)
fn init_indirect_args() {
    collision_pairs_len_indirect_args = Indirect::DispatchIndirectArgs(Indirect::div_ceil(collision_pairs_len, WORKGROUP_SIZE), 1, 1);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < num_colliders; i += num_threads) {
        let pose1 = poses[i];
        let shape1 = shapes[i];

        var aabb1 = Shape::aabb(pose1, shape1);
        let prediction = 0.1;
        let dilation = Vector(prediction); // TODO: should be configurable.
        aabb1.mins -= dilation;
        aabb1.maxs += dilation;

        for (var j = i + 1u; j < num_colliders; j++) {
            let pose2 = poses[j];
            let shape2 = shapes[j];
            let aabb2 = Shape::aabb(pose2, shape2);

            if Aabb::check_intersection(aabb1, aabb2) {
                let target_pair_index = atomicAdd(&collision_pairs_len, 1u);
                collision_pairs[target_pair_index] = vec2(i, j);
            }
        }
    }
}