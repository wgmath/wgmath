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

@group(0) @binding(6)
var<storage, read_write> domain_aabb: Aabb::Aabb;
@group(0) @binding(7)
var<storage, read_write> morton_keys: array<u32>;
@group(0) @binding(8)
var<storage, read_write> sorted_colliders: array<u32>;
@group(0) @binding(9)
var<storage, read_write> tree: array<LbvhNode>;

struct LbvhNode {
    aabb: Aabb::Aabb,
    left: u32,
    right: u32,
    parent: u32,
    refit_count: atomic<u32>, // Number of threads that reached this node for refitting in [0..2].
}

const WORKGROUP_SIZE: u32 = 64;
// NOTE: if this const is modified, don’t forget to adjust accordingly
//       the number of calls to `reduce` in `compute_domain`.
const REDUCTION_WORKGROUP_SIZE: u32 = 128;

// NOTE: the workspaces are used only by `compute_domain` for the min/max reduction.
var<workgroup> workspace_mins: array<vec3<f32>, REDUCTION_WORKGROUP_SIZE>;
var<workgroup> workspace_maxs: array<vec3<f32>, REDUCTION_WORKGROUP_SIZE>;

fn reduce(thread_id: u32, stride: u32) {
    if thread_id < stride {
        workspace_mins[thread_id] = min(workspace_mins[thread_id], workspace_mins[thread_id + stride]);
        workspace_maxs[thread_id] = max(workspace_maxs[thread_id], workspace_maxs[thread_id + stride]);
    }
    workgroupBarrier();
}

/// Runs a reduction to compute the AABB of the collider positions.
/// Needs to be called with a single workgroup.
@compute @workgroup_size(REDUCTION_WORKGROUP_SIZE, 1, 1)
fn compute_domain(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let thread_id = invocation_id.x;
    workspace_mins[thread_id] = vec3(1.0e20, 1.0e20, 1.0e20);
    workspace_maxs[thread_id] = -vec3(1.0e20, 1.0e20, 1.0e20);

    for (var i = thread_id; i < num_colliders; i += REDUCTION_WORKGROUP_SIZE) {
        let val_i = poses[i].translation_scale.xyz;
        workspace_mins[thread_id] = min(workspace_mins[thread_id], val_i);
        workspace_maxs[thread_id] = max(workspace_maxs[thread_id], val_i);
    }

    workgroupBarrier();

    reduce(thread_id, 64u);
    reduce(thread_id, 32u);
    reduce(thread_id, 16u);
    reduce(thread_id, 8u);
    reduce(thread_id, 4u);
    reduce(thread_id, 2u);
    reduce(thread_id, 1u);

    if thread_id == 0u {
        domain_aabb.mins = workspace_mins[0];
        domain_aabb.maxs = workspace_maxs[0];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn compute_morton(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // NOTE: for simplicity we compute the morton key of the collider position instead of
    //       the collider shape’s AABB center. We might want to revisit that in the future
    //       once we start adding more complex shapes.
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;

    for (var i = invocation_id.x; i < num_colliders; i += num_threads) {
        let center = poses[i].translation_scale.xyz;
        let normalized = (center - domain_aabb.mins) / (domain_aabb.maxs - domain_aabb.mins);
        let morton = morton3D(normalized.x, normalized.y, normalized.z);
        morton_keys[i] = morton;
    }
}

/// Builds each node of the tree in parallel.
///
/// This only computes the tree topology (children and parent pointers).
/// This doesn’t update the bounding boxes. Call `refit` for updating bounding boxes!
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn build(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let num_internal_nodes = num_colliders - 1u;
    let first_leaf_id = num_internal_nodes;

    for (var i = invocation_id.x; i < num_internal_nodes; i += num_threads) {
        // Determine the direction of the range (+1 or -1).
        let ii = i32(i);
        let curr_key = morton_keys[i];
        let d = sign(prefix_len(curr_key, ii + 1) - prefix_len(curr_key, ii - 1));

        // Compute upper bound for the length of the range.
        let delta_min = prefix_len(curr_key, ii - d);
        var lmax = 2; // TODO PERF: start at 128 ?

        while prefix_len(curr_key, ii + lmax * d) > delta_min {
            lmax *= 2; // TODO PERF: multiply by 4 instead of 2 ?
        }

        // Find the other end using binary search.
        var l = 0;
        for (var t = lmax / 2; t >= 1; t /= 2) {
            if prefix_len(curr_key, ii + (l + t) * d) > delta_min {
                l += t;
            }
        }
        let j = ii + l * d;

        // Find the split position using binary search.
        let delta_node = prefix_len(curr_key, j);
        var s = 0;
        var t = div_ceil(l, 2);
        loop {
            if prefix_len(curr_key, ii + (s + t) * d) > delta_node {
                s += t;
            }

            if t == 1 {
                break;
            }

            t = div_ceil(t, 2);
        }

        let gamma  = ii + s * d + min(d, 0);

        // Output child and parent pointers.
        let left = select(gamma, i32(first_leaf_id) + gamma, min(ii, j) == gamma);
        let right = select(gamma + 1, i32(first_leaf_id) + gamma + 1, max(ii, j) == gamma + 1);
        tree[i].left = u32(left);
        tree[i].right = u32(right);
        tree[i].refit_count = 0u; // Might as well reset the refit count here.
        tree[left].parent = i;
        tree[right].parent = i;
    }
}

fn prefix_len(curr_key: u32, other_index: i32) -> i32 {
    let other_key = morton_at(other_index);
    return countLeadingZeros(i32(curr_key) ^ other_key);
}

fn morton_at(i: i32) -> i32 {
    // TODO PERF: would it be meaningful to add sentinels at the begining
    //            and end of the morton_keys array so we don’t have to check
    //            bounds?
    if i < 0 || i > i32(num_colliders) - 1 {
        return -1;
    } else {
        return i32(morton_keys[u32(i)]);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn refit_leaves(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let first_leaf_id = num_colliders - 1;

    for (var i = invocation_id.x; i < num_colliders; i += num_threads) {
        let curr_leaf_id = first_leaf_id + i;
        let leaf_collider = sorted_colliders[i];
        let leaf_pose = poses[leaf_collider];
        let leaf_shape = shapes[leaf_collider];

        tree[curr_leaf_id].aabb = Aabb::from_shape(leaf_pose, leaf_shape);
        tree[curr_leaf_id].left = leaf_collider;
    }
}

// FIXME: the browser might complain about non-uniform control flow
//        because of the `workgroupBarrier` here…
// NOTE PERF: this function runs the refit with a single workgroup to work around the lack
//            of strong memory ordering semantic in WGSL for the `atomicAdd`. If we could
//            have non-relaxed ordering, we could just use `refit` instead.
@compute @workgroup_size(256, 1, 1)
fn refit_internal(@builtin(local_invocation_id) local_id: vec3<u32>) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    let num_threads = 256u;
    let first_leaf_id = num_colliders - 1;

    for (var i = local_id.x; i < num_colliders; i += num_threads) {
        let curr_leaf_id = first_leaf_id + i;
        var curr_id = tree[curr_leaf_id].parent;

        loop {
            let refit_count = atomicAdd(&tree[curr_id].refit_count, 1u);

            if refit_count == 0 {
                // If `refit_count` was 0 then the other thread hasn’t reached this node
                // yet and the sibbling aabb might not be available yet.
                // Stop the propagation to the parents here, the other thread will do it.
                break;
            }

            // If `refit_count` was 1 then the other thread has already reached this node
            // and we know the sibblings aabb is available. So we can continue the propagation.

            // TODO PERF: instead of re-reading both aabbs, we could keep the aabb from the
            //            previous loop so we don’t have to re-fetch one of the two aabbs.
            let left = tree[tree[curr_id].left].aabb;
            let right = tree[tree[curr_id].right].aabb;
            tree[curr_id].aabb = Aabb::merge(left, right);

            if curr_id == 0 {
                // We reached the root, can’t go higher.
                break;
            }

            curr_id = tree[curr_id].parent;
            workgroupBarrier();
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn refit(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let first_leaf_id = num_colliders - 1;

    for (var i = invocation_id.x; i < num_colliders; i += num_threads) {
        let curr_leaf_id = first_leaf_id + i;
        let leaf_collider = sorted_colliders[i];
        let leaf_pose = poses[leaf_collider];
        let leaf_shape = shapes[leaf_collider];

        tree[curr_leaf_id].aabb = Aabb::from_shape(leaf_pose, leaf_shape);
        tree[curr_leaf_id].left = leaf_collider;

        // Propagate to ancestors.
        var curr_id = tree[curr_leaf_id].parent;

        loop {
            let refit_count = atomicAdd(&tree[curr_id].refit_count, 1u);

            if refit_count == 0 {
                // If `refit_count` was 0 then the other thread hasn’t reached this node
                // yet and the sibbling aabb might not be available yet.
                // Stop the propagation to the parents here, the other thread will do it.
                break;
            }

            // If `refit_count` was 1 then the other thread has already reached this node
            // and we know the sibblings aabb is available. So we can continue the propagation.

            // TODO PERF: instead of re-reading both aabbs, we could keep the aabb from the
            //            previous loop so we don’t have to re-fetch one of the two aabbs.
            let left = tree[tree[curr_id].left].aabb;
            let right = tree[tree[curr_id].right].aabb;
            tree[curr_id].aabb = Aabb::merge(left, right);

            if curr_id == 0 {
                // We reached the root, can’t go higher.
                break;
            }

            curr_id = tree[curr_id].parent;
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn find_collision_pairs(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let first_leaf_id = num_colliders - 1u;

    for (var leaf_i = invocation_id.x; leaf_i < num_colliders; leaf_i += num_threads) {
        let i = tree[first_leaf_id + leaf_i].left;
        var aabb1 = tree[first_leaf_id + leaf_i].aabb;
        let prediction = 2.0e-3; // TODO: should be configurable.
        let dilation = vec3(prediction);
        aabb1.mins -= dilation;
        aabb1.maxs += dilation;

        // Traverse the tree.
        var curr_id = 0u;
        var stack = array<u32, 64>();
        var stack_len = 1u;
        stack[0u] = 0u;

        while stack_len != 0 {
            stack_len -= 1u;
            let curr_id = stack[stack_len];
            let node = &tree[curr_id];

            if curr_id >= first_leaf_id {
                // We reached a leaf, register a collision pair.
                // FIXME: we can have multiple primitives associated to the same leaf if they
                //        were given the same morton key?
                let j = (*node).left;
                // NOTE: we don’t have to compare i < j to avoid duplicates since that comparison already happened
                //       alongside the AABB check.
                let target_pair_index = atomicAdd(&collision_pairs_len, 1u);
                collision_pairs[target_pair_index] = vec2(i, j);
            } else {
                let left = (*node).left;
                let right = (*node).right;

                // Go on the child only if the AABB intersects and either the child isn’t a leaf, or it is a leaf with associated collider
                // smaller than `i` (to avoid duplicate pairs).
                if (left < first_leaf_id || i < tree[left].left) && Aabb::check_intersection(aabb1, tree[left].aabb) {
                    stack[stack_len] = (*node).left;
                    stack_len += 1u;
                }

                // NOTE: on leaves (including tree[right]), the collider id is stored as the left child index.
                if (right < first_leaf_id || i < tree[right].left) && Aabb::check_intersection(aabb1, tree[right].aabb) {
                    stack[stack_len] = (*node).right;
                    stack_len += 1u;
                }
            }
        }
    }
}

@compute @workgroup_size(1, 1, 1)
fn reset_collision_pairs() {
    collision_pairs_len = 0u;
}

@compute @workgroup_size(1, 1, 1)
fn init_indirect_args() {
    collision_pairs_len_indirect_args = Indirect::DispatchIndirectArgs(Indirect::div_ceil(collision_pairs_len, WORKGROUP_SIZE), 1, 1);
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
fn expandBits(v: u32) -> u32
{
    var vv = (v * 0x00010001u) & 0xFF0000FFu;
    vv = (vv * 0x00000101u) & 0x0F00F00Fu;
    vv = (vv * 0x00000011u) & 0xC30C30C3u;
    vv = (vv * 0x00000005u) & 0x49249249u;
    return vv;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
fn morton3D(x: f32, y: f32, z: f32) -> u32
{
    let scaled_x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    let scaled_y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    let scaled_z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    let xx = expandBits(u32(scaled_x));
    let yy = expandBits(u32(scaled_y));
    let zz = expandBits(u32(scaled_z));
    return xx * 4 + yy * 2 + zz;
}

fn div_ceil(x: i32, y: i32) -> i32 {
    return (x + y - 1) / y;
}