#import wgparry::shape as Shape;
#import wgparry::bounding_volumes::aabb as Aabb;
#import wgebra::sim3 as Pose;
#import wgcore::indirect as Indirect;

@group(0) @binding(0)
var<uniform> num_colliders: u32;
@group(0) @binding(1)
var<storage, read> poses: array<Pose::Sim3>;
@group(0) @binding(2)
var<storage, read> shapes: array<Shape::Shape>;
@group(0) @binding(3)
var<storage, read_write> collision_pairs: array<vec2<u32>>;
@group(0) @binding(4)
var<storage, read_write> collision_pairs_len: atomic<u32>;
@group(0) @binding(5)
var<storage, read_write> collision_pairs_len_indirect_args: Indirect::DispatchIndirectArgs;

@group(1) @binding(0)
var<storage, read_write> debug_mins: array<vec3<f32>>;
@group(1) @binding(1)
var<storage, read_write> debug_maxs: array<vec3<f32>>;


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
        var aabb1 = Aabb::from_shape(pose1, shape1);
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

        var aabb1 = Aabb::from_shape(pose1, shape1);
        let prediction = 0.1;
        let dilation = vec3(prediction); // TODO: should be configurable.
        aabb1.mins -= dilation;
        aabb1.maxs += dilation;

        for (var j = i + 1u; j < num_colliders; j++) {
            let pose2 = poses[j];
            let shape2 = shapes[j];
            let aabb2 = Aabb::from_shape(pose2, shape2);

            if Aabb::check_intersection(aabb1, aabb2) {
                let target_pair_index = atomicAdd(&collision_pairs_len, 1u);
                collision_pairs[target_pair_index] = vec2(i, j);
            }
        }
    }
}