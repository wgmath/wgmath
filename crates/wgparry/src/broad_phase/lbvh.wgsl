//! Linear Bounding Volume Hierarchy (LBVH) Broad Phase
//!
//! This module implements a GPU-based LBVH construciton and traversal. It is based on
//! the paper: https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf
//! The LBVH provides O(n log n) construction and O(n log n) query complexity.
//!
//! Pipeline stages:
//! 1. compute_domain: Parallel reduction to find AABB of all collider positions.
//! 2. compute_morton: Assigns 30-bit (3D) or 20-bit (2D) Morton codes to colliders.
//! 3. [External radix sort]: Sorts colliders by Morton code.
//! 4. build: Constructs binary tree topology in parallel (Karras algorithm).
//! 5. refit_leaves: Computes leaf AABBs from shapes.
//! 6. refit_internal: Bottom-up AABB propagation using atomic synchronization.
//! 7. find_collision_pairs: Tree traversal for each collider to find intersections.
//!
//! Data structures:
//! - LbvhNode: Binary tree node with AABB + left/right/parent pointers + refit counter
//! - Tree layout: Internal nodes [0..n-1], leaves [n..2n-1]
//! - Leaf nodes store collider index in 'left' field
//!
//! Key algorithms:
//! - Karras 2012: Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
//! - Atomic refitting: Each leaf spawns thread that propagates upward, atomics ensure parent
//!   waits for both children
//! - Stack-based traversal: 64-entry stack for tree queries
//!
//! Morton encoding:
//! - 3D: 10 bits per axis, interleaved (xxx...yyy...zzz -> xyzxyz...)
//! - 2D: 10 bits per axis (TODO: could be extended to 16+16)
//!
//! Performance:
//! - Construction: O(n log n)
//! - Queries: O(n log n) total (O(log n) per collider on average)
//!
//! Implementation notes:
//! - Web compatibility: Uses uniform control flow version of refit_internal
//! - Native version (without uniform control flow for atomics) available via NATIVE=1 define

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

@group(0) @binding(6)
var<storage, read_write> domain_aabb: Aabb::Aabb;
@group(0) @binding(7)
var<storage, read_write> morton_keys: array<u32>;
@group(0) @binding(8)
var<storage, read_write> sorted_colliders: array<u32>;
@group(0) @binding(9)
var<storage, read_write> tree: array<LbvhNode>;

/// A node in the Linear BVH tree.
///
/// The tree has n-1 internal nodes and n leaf nodes. Internal nodes are stored
/// in indices [0..n-1], leaf nodes in [n..2n].
///
/// For internal nodes:
/// - left/right point to child nodes (may be internal or leaf)
/// - parent points to parent internal node (0 for root)
/// - refit_count tracks how many children have updated this node's AABB
///
/// For leaf nodes:
/// - left stores the collider index
/// - right is unused
/// - parent points to parent internal node
/// - refit_count is unused
struct LbvhNode {
    /// Axis-aligned bounding box for this node's subtree.
    aabb: Aabb::Aabb,
    /// Left child index (internal) or collider index (leaf).
    left: u32,
    /// Right child index (internal nodes only).
    right: u32,
    /// Parent node index.
    parent: u32,
    /// Atomic counter for bottom-up refitting (0, 1, or 2).
    /// When a thread arrives at a node, it atomically increments this.
    /// If the old value was 0, the thread stops (sibling hasn't arrived yet).
    /// If the old value was 1, the thread continues upward (both children ready).
    refit_count: atomic<u32>,
}

const WORKGROUP_SIZE: u32 = 64;
// NOTE: if this const is modified, don’t forget to adjust accordingly
//       the number of calls to `reduce` in `compute_domain`.
const REDUCTION_WORKGROUP_SIZE: u32 = 128;

// NOTE: the workspaces are used only by `compute_domain` for the min/max reduction.
var<workgroup> workspace_mins: array<Vector, REDUCTION_WORKGROUP_SIZE>;
var<workgroup> workspace_maxs: array<Vector, REDUCTION_WORKGROUP_SIZE>;

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
    workspace_mins[thread_id] = Vector(1.0e20);
    workspace_maxs[thread_id] = -Vector(1.0e20);

    for (var i = thread_id; i < num_colliders; i += REDUCTION_WORKGROUP_SIZE) {
#if DIM == 2
        let val_i = poses[i].translation;
#else
        let val_i = poses[i].translation_scale.xyz;
#endif
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
#if DIM == 2
        let center = poses[i].translation;
#else
        let center = poses[i].translation_scale.xyz;
#endif

        let normalized = (center - domain_aabb.mins) / (domain_aabb.maxs - domain_aabb.mins);
        let morton_key = morton(normalized);
        morton_keys[i] = morton_key;
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
        let d = sign(prefix_len(curr_key, ii, ii + 1) - prefix_len(curr_key, ii, ii - 1));

        // Compute upper bound for the length of the range.
        let delta_min = prefix_len(curr_key, ii, ii - d);
        var lmax = 2; // TODO PERF: start at 128 ?

        while prefix_len(curr_key, ii, ii + lmax * d) > delta_min {
            lmax *= 2; // TODO PERF: multiply by 4 instead of 2 ?
        }

        // Find the other end using binary search.
        var l = 0;
        for (var t = lmax / 2; t >= 1; t /= 2) {
            if prefix_len(curr_key, ii, ii + (l + t) * d) > delta_min {
                l += t;
            }
        }
        let j = ii + l * d;

        // Find the split position using binary search.
        let delta_node = prefix_len(curr_key, ii, j);
        var s = 0;
        var t = div_ceil(l, 2);

        loop {
            if prefix_len(curr_key, ii, ii + (s + t) * d) > delta_node {
                s += t;
            }

            if t <= 1 {
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

fn prefix_len(curr_key: u32, curr_index: i32, other_index: i32) -> i32 {
    if other_index < 0 || other_index > i32(num_colliders) - 1 {
        return -1;
    }

    let other_key = morton_at(other_index);
    let morton_prefix_len = countLeadingZeros(i32(curr_key) ^ other_key);
    // Fallback to indices if the morton keys are equal.
    let fallback_prefix_len = 32 + countLeadingZeros(curr_index ^ other_index);
    return select(fallback_prefix_len, morton_prefix_len, i32(curr_key) != other_key);
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

        tree[curr_leaf_id].aabb = Shape::aabb(leaf_pose, leaf_shape);
        tree[curr_leaf_id].left = leaf_collider;
    }
}

// TODO: benchmark to see if we shouldn’t just use the non-native version.
#if NATIVE == 1
// FIXME: original version that doesn’t work on the web because control flow
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
            } else {
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
            }

            curr_id = tree[curr_id].parent;
            workgroupBarrier();
        }
    }
}
#else
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

    // All threads must execute the same number of outer loop iterations for uniform control flow.
    let num_iterations = (num_colliders + num_threads - 1u) / num_threads;

    for (var iter = 0u; iter < num_iterations; iter++) {
        let i = local_id.x + iter * num_threads;
        var thread_is_active = i < num_colliders;

        var curr_id = 0u;
        if thread_is_active {
            let curr_leaf_id = first_leaf_id + i;
            curr_id = tree[curr_leaf_id].parent;
        }

        // Process the tree level by level with uniform barriers.
        // Maximum tree depth is log2(num_colliders), but we use 32 as a safe upper bound.
        for (var level = 0u; level < 32u; level++) {
            if thread_is_active {
                let refit_count = atomicAdd(&tree[curr_id].refit_count, 1u);

                if refit_count == 0u {
                    // If `refit_count` was 0 then the other thread hasn't reached this node
                    // yet and the sibbling aabb might not be available yet.
                    // Stop the propagation to the parents here, the other thread will do it.
                    thread_is_active = false;
                } else {
                    // If `refit_count` was 1 then the other thread has already reached this node
                    // and we know the sibblings aabb is available. So we can continue the propagation.

                    // TODO PERF: instead of re-reading both aabbs, we could keep the aabb from the
                    //            previous loop so we don't have to re-fetch one of the two aabbs.
                    let left = tree[tree[curr_id].left].aabb;
                    let right = tree[tree[curr_id].right].aabb;
                    tree[curr_id].aabb = Aabb::merge(left, right);

                    if curr_id == 0u {
                        // We reached the root, can't go higher.
                        thread_is_active = false;
                    } else {
                        curr_id = tree[curr_id].parent;
                    }
                }
            }

            // Barrier ensures all AABB writes are complete before the next iteration's atomics.
            workgroupBarrier();
        }
    }
}
#endif

// NOTE PERF: this function runs the refit with a single workgroup to work around the lack
//            of strong memory ordering semantic in WGSL for the `atomicAdd`. If we could
//            have non-relaxed ordering, we could just use `refit` instead.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn refit_internal_requirerings_strong_atomics(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    if invocation_id.x >= num_colliders {
        return;
    }

    let first_leaf_id = num_colliders - 1;
    let curr_leaf_id = first_leaf_id + invocation_id.x;
    var curr_id = tree[curr_leaf_id].parent;

    // TODO: this only works if we could use strong memory ordering for the atomicAdd.
    //       Otherwise there is no guarantee the counter isn’t incremented before the
    //       updated aabb is written into `tree[curr_id].aabb`.
    loop {
        let refit_count = atomicAdd(&tree[curr_id].refit_count, 1u);

        if refit_count == 0 {
            // If `refit_count` was 0 then the other thread hasn’t reached this node
            // yet and the sibbling aabb might not be available yet.
            // Stop the propagation to the parents here, the other thread will do it.
            break;
        } else {
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
            // workgroupBarrier();
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

        tree[curr_leaf_id].aabb = Shape::aabb(leaf_pose, leaf_shape);
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
        let dilation = Vector(prediction);
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
                let j = (*node).left;
                // NOTE: we don’t have to compare i < j to avoid duplicates since that comparison already happened
                //       alongside the AABB check.
                let target_pair_index = atomicAdd(&collision_pairs_len, 1u);

                // NOTE: if the index is out-of-bounds (meaning the `collision_pairs` isn’t
                //       big enough), don’t write. But keep traversing so we get the exact count we need
                //       for reallocating the buffers.
                if target_pair_index < arrayLength(&collision_pairs) {
                    collision_pairs[target_pair_index] = vec2(i, j);
                }
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

#if DIM == 2
// Expands a 16-bit integer into 32 bits
// by inserting 1 zero after each bit.
fn expandBits(v: u32) -> u32
{
    var x = v & 0x0000ffffu;
    x = (x | (x << 8)) & 0x00ff00ffu;
    x = (x | (x << 4)) & 0x0f0f0f0fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}

// Calculates a 32-bit Morton code for the
// given 2D point located within the unit square [0,1].
fn morton(v: vec2<f32>) -> u32
{
    let scaled_x = clamp(v.x * 65536.0f, 0.0f, 65535.0f);
    let scaled_y = clamp(v.y * 65536.0f, 0.0f, 65535.0f);
    let xx = expandBits(u32(scaled_x));
    let yy = expandBits(u32(scaled_y));
    return xx | (yy << 1);
}
#else
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
fn morton(v: vec3<f32>) -> u32
{
    let scaled_x = min(max(v.x * 1024.0f, 0.0f), 1023.0f);
    let scaled_y = min(max(v.y * 1024.0f, 0.0f), 1023.0f);
    let scaled_z = min(max(v.z * 1024.0f, 0.0f), 1023.0f);
    let xx = expandBits(u32(scaled_x));
    let yy = expandBits(u32(scaled_y));
    let zz = expandBits(u32(scaled_z));
    return xx * 4 + yy * 2 + zz;
}
#endif

fn div_ceil(x: i32, y: i32) -> i32 {
    return (x + y - 1) / y;
}