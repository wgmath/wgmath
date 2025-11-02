//! Graph coloring for parallel constraint solving
//!
//! This shader implements graph coloring algorithms to enable parallel constraint
//! solving. The goal is to assign colors to constraints such that no two constraints
//! of the same color share a rigid body, allowing parallel solving within each color.
//!
//! Why Graph Coloring?
//! Sequential constraint solvers (Gauss-Seidel) are inherently serial because
//! solving one constraint affects bodies involved in other constraints. Graph
//! coloring breaks this dependency: constraints with the same color are independent
//! and can be solved in parallel.
//!
//! Algorithms Implemented:
//!
//! 1. Jones-Plassmann-Luby (Luby's Algorithm):
//!    - Randomized parallel algorithm.
//!    - Each iteration: nodes with locally maximal random weights are colored.
//!    - Better for sparse conflict graphs.
//!    - Reference: https://developer.nvidia.com/blog/graph-coloring-more-parallelism-for-incomplete-lu-factorization/
//!
//! 2. Topo-GC (Topological Graph Coloring):
//!    - Each node selects the smallest available color not used by neighbors.
//!    - Typically produces fewer colors than Luby.
//!    - Better for dense conflict graphs.
//!    - Reference: https://people.csail.mit.edu/xchen/docs/ipdpsw-2016.pdf
//!
//! Performance:
//! - Workgroup size: 64 threads (Topo-GC uses 1 for some kernels)
//! - Luby: O(log n) iterations (average, O(n) worst-case).
//! - Topo-GC: O(d) iterations where d is max degree, fewer colors
//!
//! Color Limits:
//! - Topo-GC: Up to 63 colors (using 2x u32 bitmasks with 1 reserved bit for bookkeeping)
//! - Luby: Unlimited colors

#import wgrapier::dynamics::constraint as Constraint


/// Prefix sum of constraint counts per body.
/// Used to find the range [counts[i-1], counts[i]) of constraints for body i.
@group(0) @binding(0)
var<storage, read> body_constraint_counts: array<u32>;
@group(0) @binding(1)
var<storage, read> body_constraint_ids: array<u32>;
// TODO: instead of reading from the constraint, would it be more efficient to store the second body id involved
//       in the constraint in a separate buffer?
@group(0) @binding(2)
var<storage, read> constraints: array<Constraint::TwoBodyConstraint>;
@group(0) @binding(3)
var<storage, read_write> constraints_colors: array<u32>;
@group(0) @binding(4)
var<storage, read_write> constraints_rands: array<u32>;
@group(0) @binding(5)
var<uniform> curr_color: u32;
@group(0) @binding(6)
var<storage, read_write> uncolored: atomic<u32>;
@group(0) @binding(7)
var<storage, read> contacts_len: u32;

@group(1) @binding(0)
var<storage, read_write> num_colors: atomic<u32>;
@group(1) @binding(1)
var<storage, read_write> colored: array<u32>;

const WORKGROUP_SIZE: u32 = 64;
const MAX_U32: u32 = 4294967295;

/*
 * Jones-Plassmann-Luby Graph Coloring Algorithm
 *
 * Randomized parallel graph coloring algorithm. In each iteration:
 * 1. Each uncolored node compares its random weight with neighbors
 * 2. Nodes with locally maximal weights are colored
 * 3. Repeat until all nodes are colored
 *
 * Expected iterations: O(log n) for bounded-degree graphs
 * Expected colors: O(Δ) where Δ is maximum degree
 *
 * This implementation uses a hash function instead of a true RNG for
 * simplicity.
 */

/// Hash function for generating random weights.
///
/// Uses a variant of the Murmur3 hash function to generate pseudo-random
/// weights from constraint indices.
///
/// @param packed_key: Input value (constraint index)
/// @returns: Pseudo-random hash value
fn hash(packed_key: u32) -> u32 {
    var key = packed_key;
    key *= 0xcc9e2d51u;
    key = (key << 15) | (key >> 17);
    key *= 0x1b873593u;
    return key;
}

/// Initializes Luby algorithm state.
///
/// Sets all constraints to uncolored (MAX_U32) and assigns random weights
/// using the hash function.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn reset_luby(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
) {
    let i = invocation_id.x;

    if i < contacts_len {
        // Mark as uncolored
        constraints_colors[i] = MAX_U32;

        // Assign random weight
        constraints_rands[i] = hash(i);
    }
}

/// Performs one iteration of Luby's graph coloring algorithm.
///
/// For each uncolored constraint:
/// 1. Compare random weight with all neighboring constraints
/// 2. If this constraint has the maximum weight among uncolored neighbors,
///    assign it the current color
/// 3. Otherwise, leave it uncolored for the next iteration
///
/// This kernel is called repeatedly, incrementing curr_color each time,
/// until all constraints are colored.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn step_graph_coloring_luby(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var constraint_i = invocation_id.x; constraint_i < contacts_len; constraint_i += num_threads) {
        if constraints_colors[constraint_i] != MAX_U32 {
            // This constraint already has a color.
            continue;
        }

        let rand_i = constraints_rands[constraint_i];
        let body_a = constraints[constraint_i].solver_body_a;
        let body_b = constraints[constraint_i].solver_body_b;

        var first_constraint_id_a = 0u;
        let last_constraint_id_a = body_constraint_counts[body_a];
        if body_a != 0u {
            first_constraint_id_a = body_constraint_counts[body_a - 1u];
        }

        var first_constraint_id_b = 0u;
        let last_constraint_id_b = body_constraint_counts[body_b];
        if body_b != 0u {
            first_constraint_id_b = body_constraint_counts[body_b - 1u];
        }

        var is_greatest = true;

        // Traverse all constraints from body A.
        for (var j = first_constraint_id_a; is_greatest && j < last_constraint_id_a; j += 1u) {
            let constraint_j = body_constraint_ids[j];
            let rand_j = constraints_rands[constraint_j];
            let color_j = constraints_colors[constraint_j];
            // NOTE: there is a very rare case both constraints got assigned the same random number.
            //       in that case, we define the "greatest" comparison based on the constraint’s array index.
            // NOTE: the equality in i >= j is important here to account for the fact we will iterate
            //       through the current constraint’s index too.
            is_greatest = is_greatest && (color_j != MAX_U32 || rand_i > rand_j || (rand_i == rand_j && constraint_i >= constraint_j));
        }
        // Traverse all constraints from body B.
        for (var j = first_constraint_id_b; is_greatest && j < last_constraint_id_b; j += 1u) {
            let cid = body_constraint_ids[j];
            let rand_j = constraints_rands[cid];
            let color_j = constraints_colors[cid];
            // NOTE: there is a very rare case both constraints got assigned the same random number.
            //       in that case, we define the "greatest" comparison based on the constraint’s array index.
            // NOTE: the equality in i >= j is important here to account for the fact we will iterate
            //       through the current constraint’s index too.
            is_greatest = is_greatest && (color_j != MAX_U32 || rand_j < rand_i || (rand_j == rand_j && constraint_i >= cid));
        }

        if is_greatest {
            constraints_colors[constraint_i] = curr_color;
        } else {
            atomicAdd(&uncolored, 1u);
        }
    }
}

/*
 * Topo-GC (Topological Graph Coloring) Algorithm
 * ==============================================================================
 *
 * Parallel graph coloring algorithm. Each iteration:
 * 1. Each uncolored node selects the smallest color not used by neighbors
 * 2. Conflicts are detected and resolved in the next iteration
 * 3. Repeat until convergence (no conflicts)
 *
 * Reference: https: *people.csail.mit.edu/xchen/docs/ipdpsw-2016.pdf
 *
 * Advantages:
 * - Typically produces fewer colors than Luby (closer to optimal)
 * - Faster convergence for dense graphs
 *
 * Disadvantages:
 * - Limited to 63 colors (due to 2x u32 bitmask representation)
 * - May require more iterations than Luby in worst case
 *
 * Algorithm Steps:
 * 1. reset_topo_gc: Initialize all nodes as uncolored
 * 2. step_graph_coloring_topo_gc: Each node selects smallest available color
 * 3. fix_conflicts_topo_gc: Detect and uncolor conflicting nodes
 * 4. Repeat steps 2-3 until num_colors > 0 (no conflicts, algorithm finished)
 *
 * Color Representation:
 * Uses a 64-bit bitmask (2x u32) to track occupied colors for each node.
 * Bit i set means color i is used by a neighbor.
 * Color indices start at 1. (The index 0 is reserved as an implementation detail.)
 */

/// Initializes Topo-GC algorithm state.
///
/// Sets all constraints to color 0 (uncolored) and marks them as not yet colored.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn reset_topo_gc(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
) {
    let i = invocation_id.x;

    if i < contacts_len {
        // Color 0 is reserved for "uncolored" state
        constraints_colors[i] = 0u;
        colored[i] = 0u;
    }
}

/// Resets the convergence flag for Topo-GC.
///
/// Sets num_colors to 1 to indicate the algorithm should run.
/// When the algorithm completes, num_colors will be set to 0 by the coloring step.
@compute @workgroup_size(1, 1, 1)
fn reset_completion_flag_topo_gc(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
) {
    if invocation_id.x == 0u {
        // Non-zero value indicates algorithm should continue
        num_colors = 1u;
    }
}

/// Performs one iteration of Topo-GC coloring.
///
/// For each uncolored constraint:
/// 1. Build a bitmask of colors used by neighboring constraints
/// 2. Select the smallest color (lowest bit) not in the bitmask
/// 3. Mark this constraint as colored
///
/// Uses a 64-bit bitmask (2x u32) to track up to 63 colors (color 0 = uncolored).
/// countTrailingZeros finds the position of the first unset bit, giving the
/// smallest available color.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn step_graph_coloring_topo_gc(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var constraint_i = invocation_id.x; constraint_i < contacts_len; constraint_i += num_threads) {
        if colored[constraint_i] != 0u {
            // This constraint already has a color.
            continue;
        }

        // NOTE: generates up to 63 colors.
        // Note that we always mark the color 0 as occupied (cf. paper using i > 0).
        var color_mask = vec2(1u, 0u);

        let body_a = constraints[constraint_i].solver_body_a;
        let body_b = constraints[constraint_i].solver_body_b;

        var first_constraint_id_a = 0u;
        let last_constraint_id_a = body_constraint_counts[body_a];
        if body_a != 0u {
            first_constraint_id_a = body_constraint_counts[body_a - 1u];
        }

        var first_constraint_id_b = 0u;
        let last_constraint_id_b = body_constraint_counts[body_b];
        if body_b != 0u {
            first_constraint_id_b = body_constraint_counts[body_b - 1u];
        }

        // Traverse all constraints from body A.
        for (var j = first_constraint_id_a; j < last_constraint_id_a; j += 1u) {
            let constraint_j = body_constraint_ids[j];

            if constraint_j == constraint_i {
                continue;
            }

            let color_j = constraints_colors[constraint_j];
            if color_j < 32u {
                color_mask.x = color_mask.x | (1u << color_j);
            } else {
                color_mask.y = color_mask.y | (1u << (color_j - 32u));
            }
        }

        // Traverse all constraints from body B.
        for (var j = first_constraint_id_b; j < last_constraint_id_b; j += 1u) {
            let constraint_j = body_constraint_ids[j];

            if constraint_j == constraint_i {
                continue;
            }

            let color_j = constraints_colors[constraint_j];
            if color_j < 32u {
                color_mask.x = color_mask.x | (1u << color_j);
            } else {
                color_mask.y = color_mask.y | (1u << (color_j - 32u));
            }
        }

        let my_color = countTrailingZeros(~color_mask.x) + countTrailingZeros(~color_mask.y);
        constraints_colors[constraint_i] = my_color;
        colored[constraint_i] = 1u;
        num_colors = 0u; // We are not finished coloring. 0 indicates the algorithm must continue.
    }
}


@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn fix_conflicts_topo_gc(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var constraint_i = invocation_id.x; constraint_i < contacts_len; constraint_i += num_threads) {
        let color_i = constraints_colors[constraint_i];

        // NOTE: this `num_colors` read doesn’t need to be atomic. Any non-zero value is indicative of a finished
        //       algorithm.
        // If `num_colors > 0u` then we know that the coloring algorithm has converged. So we use this dispatch
        // as an opportunity to compute the colors count that will be ready back to the CPU side.
        if num_colors > 0u {
            // TODO PERF: not sure if that would have a significant impact but we could keep track of
            //            whether the last iteration of the TOPO-GC algorithm already finished, in which
            //            case we can skip the atomic max entirely and just early-exist.

            atomicMax(&num_colors, color_i);
            continue;
        }

        let body_a = constraints[constraint_i].solver_body_a;
        let body_b = constraints[constraint_i].solver_body_b;

        var first_constraint_id_a = 0u;
        let last_constraint_id_a = body_constraint_counts[body_a];
        if body_a != 0u {
            first_constraint_id_a = body_constraint_counts[body_a - 1u];
        }

        var first_constraint_id_b = 0u;
        let last_constraint_id_b = body_constraint_counts[body_b];
        if body_b != 0u {
            first_constraint_id_b = body_constraint_counts[body_b - 1u];
        }

        // Traverse all constraints from body A.
        for (var j = first_constraint_id_a; j < last_constraint_id_a; j += 1u) {
            let constraint_j = body_constraint_ids[j];

            if constraint_j == constraint_i {
                continue;
            }

            let color_j = constraints_colors[constraint_j];
            if color_i == color_j && constraint_i < constraint_j {
                // Found a conflict, uncolor this node.
                colored[constraint_i] = 0u;
//                constraints_colors[constraint_i] = 0u;
            }
        }

        // Traverse all constraints from body B.
        for (var j = first_constraint_id_b; j < last_constraint_id_b; j += 1u) {
            let constraint_j = body_constraint_ids[j];

            if constraint_j == constraint_i {
                continue;
            }

            let color_j = constraints_colors[constraint_j];
            if color_i == color_j && constraint_i < constraint_j {
                // Found a conflict, uncolor this node.
                colored[constraint_i] = 0u;
//                constraints_colors[constraint_i] = 0u;
            }
        }
    }
}

// TODO PERF: add a kernel to sort the constraints based on their color for better memory locality in the solver?
//            Or maybe the sort should be a by-product of the solver initialization.