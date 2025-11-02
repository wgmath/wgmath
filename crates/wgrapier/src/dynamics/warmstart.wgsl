//! Constraint warmstarting (impulse caching across frames)
//!
//! This shader implements constraint warmstarting to transfer impulses solved at step `n - 1`
//! to the constraint waiting to be solved at step `n`.
//!
//! What is Warmstarting?
//! Instead of starting each frame's constraint solving from zero impulses, we
//! reuse impulses from the previous frame as initial guesses. This dramatically
//! reduces the number of iterations needed for convergence.
//!
//! Why Warmstarting Works:
//! - Physics simulations have temporal coherence: contacts from one frame are
//!   likely to persist to the next frame with similar impulse magnitudes
//! - Starting from a good initial guess (previous frame's solution) means fewer
//!   iterations to reach the correct solution
//! - Improves stability by preventing sudden impulse changes between frames
//!
//! Contact Point Matching:
//! - Contacts are matched by proximity in local coordinates.
//! - Distance threshold: currently set to 10cm.
//! - This handles small movements and minor geometry changes.

#import wgrapier::dynamics::constraint as Constraint

/// Length of the contact constraints array.
@group(0) @binding(0)
var<storage, read> contacts_len: u32;
/// Prefix sum of constraint counts per body (previous frame).
/// Used to find the range of constraints for each body.
@group(0) @binding(1)
var<storage, read> old_body_constraint_counts: array<u32>;

/// Constraint IDs sorted by body (previous frame).
/// Maps from body constraint index to global constraint ID.
@group(0) @binding(2)
var<storage, read> old_body_constraint_ids: array<u32>;

/// Constraints from the previous frame (read-only).
/// Source of impulses for warmstarting.
@group(0) @binding(3)
var<storage, read> old_constraints: array<Constraint::TwoBodyConstraint>;

/// Constraint builders from the previous frame (read-only).
/// Contains local contact point positions for matching.
@group(0) @binding(4)
var<storage, read> old_constraint_builders: array<Constraint::TwoBodyConstraintBuilder>;

/// Constraints for the current frame (read-write).
/// Impulses are written here during warmstarting.
@group(0) @binding(5)
var<storage, read_write> new_constraints: array<Constraint::TwoBodyConstraint>;

/// Constraint builders for the current frame (read-only).
/// Contains local contact point positions for matching.
@group(0) @binding(6)
var<storage, read> new_constraint_builders: array<Constraint::TwoBodyConstraintBuilder>;

/// Workgroup size: 64 threads per workgroup.
const WORKGROUP_SIZE: u32 = 64;

/// Transfers warmstart impulses from previous frame to current frame.
///
/// For each new constraint:
/// 1. Identify the two bodies involved
/// 2. Search old constraints for matching body pair
/// 3. Match contact points by local position proximity
/// 4. Copy accumulated impulses if match found
///
/// Assumptions:
/// - Solver body IDs in constraints match the body array indices
/// - Body pair order is consistent across frames (A,B not swapped to B,A)
/// - Contact points don't move more than 10cm in local space
///
/// Contact Point Matching:
/// - Compares local_pt_a and local_pt_b for proximity
/// - Distance threshold: 10cm (sq_threshold = 0.01 m²)
/// - Handles small movements and rotations
///
/// @param invocation_id: Global thread ID
// NOTE: this assumes that the solver body ids in the constraints match the index of the body itself.
//       This also assumes that bodies in a given constraint pair are always in the same order (they don't
//       get swapped from one frame to another).
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn transfer_warmstart_impulses(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let cid_new = invocation_id.x;

    if cid_new < contacts_len {
        // Get the two bodies involved in this new constraint
        let body_a = new_constraints[cid_new].solver_body_a;
        let body_b = new_constraints[cid_new].solver_body_b;

        // Find the range of old constraints involving body_a
        // old_body_constraint_counts is a prefix sum, so the range is [counts[i-1], counts[i])
        var first_constraint_id_a = 0u;
        let last_constraint_id_a = old_body_constraint_counts[body_a];
        if body_a != 0u {
            first_constraint_id_a = old_body_constraint_counts[body_a - 1u];
        }

        // Find the range of old constraints involving body_b
        var first_constraint_id_b = 0u;
        let last_constraint_id_b = old_body_constraint_counts[body_b];
        if body_b != 0u {
            first_constraint_id_b = old_body_constraint_counts[body_b - 1u];
        }

        let len_a = last_constraint_id_a - first_constraint_id_a;
        let len_b = last_constraint_id_b - first_constraint_id_b;

        // Optimization: search the smaller constraint list to minimize iterations
        // Also avoid static bodies which may have zero-length lists
        // Select the smallest list with a nonzero size (for example static bodies would have
        // a zero-length list despite having some constranits).
        // TODO: compare this approach with just using a hashmap.
        let ref_body = select(body_b, body_a, len_a != 0 && len_a < len_b);
        let first_constraint_id_ref = select(first_constraint_id_a, first_constraint_id_b, ref_body == body_b);
        let last_constraint_id_ref = select(last_constraint_id_a, last_constraint_id_b, ref_body == body_b);

        // Search through old constraints for matching body pair
        for (var j = first_constraint_id_ref; j < last_constraint_id_ref; j++) {
            let cid_old = old_body_constraint_ids[j];

            // Check if this old constraint involves the same body pair
            if old_constraints[cid_old].solver_body_a == body_a && old_constraints[cid_old].solver_body_b == body_b {
                // Body pair match found! Now match individual contact points.
                // We don't have feature IDs, so matching is done by proximity in local space.

                // Distance threshold for matching contact points (10cm)
                let dist_threshold = 1.0e-1; // 10cm
                let sq_threshold = dist_threshold * dist_threshold;

                // Try to match each new contact point with old contact points
                for (var k_new = 0u; k_new < new_constraints[cid_new].len; k_new++) {
                    let pt_new_a = new_constraint_builders[cid_new].infos[k_new].local_pt_a;
                    let pt_new_b = new_constraint_builders[cid_new].infos[k_new].local_pt_b;

                    // Search through old contact points for a match
                    for (var k_old = 0u; k_old < old_constraints[cid_old].len; k_old++) {
                        let pt_old_a = old_constraint_builders[cid_old].infos[k_old].local_pt_a;
                        let pt_old_b = old_constraint_builders[cid_old].infos[k_old].local_pt_b;

                        // Compute distance between contact points in local space
                        let dpt_a = pt_old_a - pt_new_a;
                        let dpt_b = pt_old_b - pt_new_b;

                        // If both points are close enough, consider it a match
                        if dot(dpt_a, dpt_a) < sq_threshold && dot(dpt_b, dpt_b) < sq_threshold {
                            // Contact point match found! Transfer the accumulated impulse.
                            // The impulse field contains the last substep's impulse, which serves
                            // as the warmstart value for this frame.
                            // NOTE: we sum the impulse + impulse_accumulator since the accumulater contains the
                            //       accumulated impulse for all the substeps except the last one.
                            // TODO: what if we have multiple matches? (currently uses first match)
                            new_constraints[cid_new].elements[k_new].normal_part.impulse =
                                old_constraints[cid_old].elements[k_old].normal_part.impulse;
                            new_constraints[cid_new].elements[k_new].normal_part.impulse_jacobi =
                                old_constraints[cid_old].elements[k_old].normal_part.impulse_jacobi;
                            new_constraints[cid_new].elements[k_new].tangent_part.impulse =
                                old_constraints[cid_old].elements[k_old].tangent_part.impulse;
                            new_constraints[cid_new].elements[k_new].tangent_part.impulse_jacobi =
                                old_constraints[cid_old].elements[k_old].tangent_part.impulse_jacobi;
                        }
                    }
                }

                // Since we found a matching body pair, no need to search further
                break;
            }
        }
    }
}