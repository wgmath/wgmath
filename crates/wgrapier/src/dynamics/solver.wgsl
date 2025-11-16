//! Main physics solver kernels (PGS/Sequential Impulse)
//!
//! This shader contains the core physics solver implementation using iterative
//! constraint-based methods. It two methods:
//! - `Soft-TGS` (the same approach as Rapier and other engines like Box2D). This operates by splitting the simulation
//!   timestep into smaller substeps in order to lower errors caused by nonlinearities (e.g. rotations). Each substep
//!   is solved with a single PGS iteration (with bias) followed by position update, followed by another PGS iteration
//!   (without bias).
//! - `Soft-Jacobi`: this is similar to `Soft-TGS` but using a pseudo-Jacobi solver instead of PGS. It is "Jacobi-like"
//!                  because, instead of solving each constraint independently in parallel, each **body** is solved
//!                  in parallel. This means that each thread will solve all the constraints affecting a given body
//!                  independently. Note that this technically violates Newton’s third law. However, vanilla Jacobi
//!                  proved to be entirely useless (too unstable) so we made this compromise.
//! For most usages `Soft-TGS` should be used instead of `Soft-Jacobi` which is significantly less stable and much
//! more energetic.

#import wgrapier::dynamics::constraint as Constraint
#import wgrapier::dynamics::sim_params as Params
#import wgparry::contact as Contact;
#import wgrapier::body as Body;
#import wgebra::inv as Inv;

#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif


@group(0) @binding(0)
var<storage, read_write> contacts: array<Contact::IndexedManifold>;
@group(0) @binding(1)
var<storage, read_write> contacts_len: u32;
@group(0) @binding(2)
var<storage, read_write> constraints: array<Constraint::TwoBodyConstraint>;
@group(0) @binding(3)
var<storage, read_write> solver_vels: array<Body::Velocity>;
@group(0) @binding(4)
var<storage, read_write> solver_vels_out: array<Body::Velocity>;
@group(0) @binding(5)
var<storage, read_write> body_constraint_counts_atomic: array<atomic<u32>>;
@group(0) @binding(5)
var<storage, read_write> body_constraint_counts: array<u32>;
@group(0) @binding(6)
var<storage, read_write> body_constraint_ids: array<u32>;
@group(0) @binding(7)
var<storage, read> constraints_colors: array<u32>;
@group(0) @binding(8)
var<storage, read_write> curr_color: u32;

@group(1) @binding(0)
var<uniform> params: Params::SimParams;
@group(1) @binding(1)
var<storage, read_write> poses: array<Transform>;
@group(1) @binding(2)
var<storage, read_write> vels: array<Body::Velocity>;
@group(1) @binding(3)
var<storage, read> mprops: array<Body::WorldMassProperties>;
@group(1) @binding(4)
var<uniform> num_colliders: u32;
@group(1) @binding(5)
var<storage, read> local_mprops: array<Body::LocalMassProperties>;
@group(1) @binding(6)
var<storage, read_write> solver_vels_inc: array<Body::Velocity>;
@group(1) @binding(7)
var<storage, read_write> constraint_builders: array<Constraint::TwoBodyConstraintBuilder>;

const WORKGROUP_SIZE: u32 = 64;


@compute @workgroup_size(1, 1, 1)
fn reset_color() {
    // NOTE: our first colors start at 1 instead of 0.
    curr_color = 1u;
}

@compute @workgroup_size(1, 1, 1)
fn inc_color() {
    curr_color += 1;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn init_constraints(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < contacts_len; i += num_threads) {
        contact_to_constraint(i, contacts[i]);
        let body1 = contacts[i].colliders.x;
        let body2 = contacts[i].colliders.y;

        // HACK: add a better way of identifying static bodies.
        if any(mprops[body1].inv_mass != Vector()) {
            atomicAdd(&body_constraint_counts_atomic[body1], 1u);
        }

        // HACK: add a better way of identifying static bodies.
        if any(mprops[body2].inv_mass != Vector()) {
            atomicAdd(&body_constraint_counts_atomic[body2], 1u);
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn update_constraints(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < contacts_len; i += num_threads) {
        let builder = constraint_builders[i];
        let body1 = constraints[i].solver_body_a;
        let body2 = constraints[i].solver_body_b;

        let cfm_factor = Params::contact_cfm_factor(params);
        let inv_dt = Params::inv_dt(params);
        let allowed_lin_err = Params::allowed_linear_error(params);
        let erp_inv_dt = Params::contact_erp_inv_dt(params);
        let max_corrective_velocity = Params::max_corrective_velocity(params);
        let warmstart_coeff = params.warmstart_coefficient;

        let poses1 = poses[body1];
        let poses2 = poses[body2];
        let num_contacts = constraints[i].len;

#if DIM == 2
        let tangents1 = vec2(-constraints[i].dir_a.y, constraints[i].dir_a.x);
#else
        let tangents1 = array(
            constraints[i].tangent_a,
            cross(constraints[i].dir_a, constraints[i].tangent_a),
        );
#endif

        for (var j = 0u; j < num_contacts; j++)
        {
            // NOTE: the tangent velocity is equivalent to an additional movement of the first body’s surface.
            let info = constraint_builders[i].infos[j];
            let p1 = Pose::mulPt(poses1, info.local_pt_a); // TODO (conveyor belts): + info.tangent_vel * solved_dt;
            let p2 = Pose::mulPt(poses2, info.local_pt_b);
            let dist = info.dist + dot(p1 - p2, constraints[i].dir_a);

            // Normal part.
            {
                let rhs_wo_bias = info.normal_vel + max(dist, 0.0) * inv_dt;
                let rhs_bias = clamp((dist + allowed_lin_err) * erp_inv_dt, -max_corrective_velocity, 0.0);
                let new_rhs = rhs_wo_bias + rhs_bias;

                constraints[i].elements[j].normal_part.rhs_wo_bias = rhs_wo_bias;
                constraints[i].elements[j].normal_part.rhs = new_rhs;
                constraints[i].elements[j].normal_part.impulse *= warmstart_coeff;
                constraints[i].elements[j].normal_part.impulse_jacobi *= warmstart_coeff;
            }

            // tangent parts.
            {
#if DIM == 2
                constraints[i].elements[j].tangent_part.impulse[0] *= warmstart_coeff;
                constraints[i].elements[j].tangent_part.impulse_jacobi[0] *= warmstart_coeff;
                let bias = dot(p1 - p2, tangents1) * inv_dt;
                constraints[i].elements[j].tangent_part.rhs[0] = constraints[i].elements[j].tangent_part.rhs_wo_bias[0] + bias;
#else
                constraints[i].elements[j].tangent_part.impulse *= warmstart_coeff;
                constraints[i].elements[j].tangent_part.impulse_jacobi *= warmstart_coeff;
                let bias0 = dot(p1 - p2, tangents1[0]) * inv_dt;
                constraints[i].elements[j].tangent_part.rhs[0] = constraints[i].elements[j].tangent_part.rhs_wo_bias[0] + bias0;
                let bias1 = dot(p1 - p2, tangents1[1]) * inv_dt;
                constraints[i].elements[j].tangent_part.rhs[1] = constraints[i].elements[j].tangent_part.rhs_wo_bias[1] + bias1;
#endif
            }
        }

        constraints[i].cfm_factor = cfm_factor;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn sort_constraints(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < contacts_len; i += num_threads) {
        let body1 = contacts[i].colliders.x;
        let body2 = contacts[i].colliders.y;

        // HACK: add a better way of identifying static bodies.
        if any(mprops[body1].inv_mass != Vector()) {
            let id1 = atomicAdd(&body_constraint_counts_atomic[body1], 1u);
            body_constraint_ids[id1] = i;
        }

        // HACK: add a better way of identifying static bodies.
        if any(mprops[body2].inv_mass != Vector()) {
            let id2 = atomicAdd(&body_constraint_counts_atomic[body2], 1u);
            body_constraint_ids[id2] = i;
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn cleanup(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < num_colliders; i += num_threads) {
        body_constraint_counts[i] = 0;

        // HACK: to handle static bodies.
        if any(mprops[i].inv_mass != Vector()) {
            solver_vels[i].linear = vels[i].linear;
            solver_vels[i].angular = vels[i].angular;
        } else {
            solver_vels[i].linear = Vector(0.0);
            solver_vels[i].angular = AngVector(0.0);
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn step_jacobi(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;

    for (var body_id = invocation_id.x; body_id < num_colliders; body_id += num_threads) {
        var first_constraint_id = 0u;
        if body_id != 0u {
            first_constraint_id = body_constraint_counts[body_id - 1u];
        }

        let last_constraint_id = body_constraint_counts[body_id];
        var solver_vel = solver_vels[body_id];

        for (var i = first_constraint_id; i < last_constraint_id; i += 1u) {
            let cid = body_constraint_ids[i];

            let solver_body1 = constraints[cid].solver_body_a;
            let solver_body2 = constraints[cid].solver_body_b;
            let dir_a = constraints[cid].dir_a;
            let friction_coeff = constraints[cid].limit;
            let im_a = constraints[cid].im_a;
            let im_b = constraints[cid].im_b;
            let cfm_factor = constraints[cid].cfm_factor;
            var solver_vel1 = solver_vels[solver_body1];
            var solver_vel2 = solver_vels[solver_body2];

            if solver_body1 == body_id {
                solver_vel1 = solver_vel;
            } else {
                solver_vel2 = solver_vel;
            }

    #if DIM == 3
            let tangent_a = constraints[cid].tangent_a;
    #else
            let tangent_a = vec2(-dir_a.y, dir_a.x);
    #endif

            // TODO: unroll?
            for (var k = 0u; k < constraints[cid].len; k++) {
                var limit = 0.0; // Friction impulse limit.

                // Solve the normal part of the constraint.
                {
                    let c = constraints[cid].elements[k].normal_part;
                    let prev_impulse = select(c.impulse_jacobi, c.impulse, solver_body1 == body_id);
                    let dvel = dot(dir_a, solver_vel1.linear) + gdot(c.torque_dir_a, solver_vel1.angular)
                        - dot(dir_a, solver_vel2.linear)
                        + gdot(c.torque_dir_b, solver_vel2.angular)
                        + c.rhs;
                    let new_impulse = cfm_factor * max(prev_impulse - c.r * dvel, 0.0);
                    let delta_impulse = new_impulse - prev_impulse;

                    if solver_body1 == body_id {
                        constraints[cid].elements[k].normal_part.impulse = new_impulse;
                    } else {
                        constraints[cid].elements[k].normal_part.impulse_jacobi = new_impulse;
                    }

                    solver_vel1.linear += dir_a * im_a * delta_impulse;
                    solver_vel1.angular += c.ii_torque_dir_a * delta_impulse;
                    solver_vel2.linear += dir_a * im_b * -delta_impulse;
                    solver_vel2.angular += c.ii_torque_dir_b * delta_impulse;

                    limit = new_impulse * friction_coeff;
                }

                // Solve the tangent parts of the constraint.
                {
                    var c = constraints[cid].elements[k].tangent_part;

        #if DIM == 2
                    let prev_impulse = select(c.impulse_jacobi[0], c.impulse[0], solver_body1 == body_id);
                    let dvel = dot(tangent_a, solver_vel1.linear) + gdot(c.torque_dir_a[0], solver_vel1.angular)
                        - dot(tangent_a, solver_vel2.linear)
                        + gdot(c.torque_dir_b[0], solver_vel2.angular)
                        + c.rhs[0];
                    let new_impulse = cfm_factor * clamp(prev_impulse - c.r[0] * dvel, -limit, limit);
                    let delta_impulse = new_impulse - prev_impulse;

                    if solver_body1 == body_id {
                        constraints[cid].elements[k].tangent_part.impulse[0] = new_impulse;
                    } else {
                        constraints[cid].elements[k].tangent_part.impulse_jacobi[0] = new_impulse;
                    }

                    solver_vel1.linear += tangent_a * im_a * delta_impulse;
                    solver_vel1.angular += c.ii_torque_dir_a[0] * delta_impulse;
                    solver_vel2.linear += tangent_a * im_b * -delta_impulse;
                    solver_vel2.angular += c.ii_torque_dir_b[0] * delta_impulse;
        #else
                    let prev_impulse = select(c.impulse_jacobi, c.impulse, solver_body1 == body_id);
                    let tangents_a = array(tangent_a, cross(dir_a, tangent_a));
                    let dvel_0 = dot(tangents_a[0], solver_vel1.linear)
                        + gdot(c.torque_dir_a[0], solver_vel1.angular)
                        - dot(tangents_a[0], solver_vel2.linear)
                        + gdot(c.torque_dir_b[0], solver_vel2.angular)
                        + c.rhs[0];
                    let dvel_1 = dot(tangents_a[1], solver_vel1.linear)
                        + gdot(c.torque_dir_a[1], solver_vel1.angular)
                        - dot(tangents_a[1], solver_vel2.linear)
                        + gdot(c.torque_dir_b[1], solver_vel2.angular)
                        + c.rhs[1];

                    let dvel_00 = dvel_0 * dvel_0;
                    let dvel_11 = dvel_1 * dvel_1;
                    let dvel_01 = dvel_0 * dvel_1;
                    let inv_lhs = (dvel_00 + dvel_11)
                        * maybe_inv(
                            dvel_00 * c.r[0] + dvel_11 * c.r[1] + dvel_01 * c.r[2],
                        );
                    let delta_impulse = vec2(inv_lhs * dvel_0, inv_lhs * dvel_1);
                    var new_impulse = prev_impulse - delta_impulse;
                    new_impulse = cap_magnitude(new_impulse, limit);

                    let dlambda = new_impulse - prev_impulse;

                    if solver_body1 == body_id {
                        constraints[cid].elements[k].tangent_part.impulse = new_impulse;
                    } else {
                        constraints[cid].elements[k].tangent_part.impulse_jacobi = new_impulse;
                    }

                    solver_vel1.linear +=
                        (tangents_a[0] * dlambda[0] + tangents_a[1] * dlambda[1]) * im_a;
                    solver_vel1.angular +=
                        c.ii_torque_dir_a[0] * dlambda[0] + c.ii_torque_dir_a[1] * dlambda[1];
                    solver_vel2.linear +=
                        (tangents_a[0] * -dlambda[0] + tangents_a[1] * -dlambda[1]) * im_b;
                    solver_vel2.angular +=
                        c.ii_torque_dir_b[0] * dlambda[0] + c.ii_torque_dir_b[1] * dlambda[1];
        #endif
                }
            }

            solver_vel.linear = select(solver_vel2.linear, solver_vel1.linear, solver_body1 == body_id);
            solver_vel.angular = select(solver_vel2.angular, solver_vel1.angular, solver_body1 == body_id);
        }

        solver_vels_out[body_id] = solver_vel;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn init_solver_vels_inc(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let i = invocation_id.x;

    if i < num_colliders {
        solver_vels_inc[i].linear = Vector(0.0);
        solver_vels_inc[i].angular = AngVector(0.0);

        // TODO: this isn’t a very pretty way of detecting static bodies.
        if any(mprops[i].inv_mass != Vector()) {
            // TODO: this currently only handles gravity.
            // TODO: make the gravity configurable
    #if DIM == 2
            let gravity = vec2(0.0, -9.81);
    #else
            let gravity = vec3(0.0, -9.81, 0.0);
    #endif

            solver_vels_inc[i].linear = gravity * params.dt;
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn apply_solver_vels_inc(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    if i < num_colliders {
        solver_vels[i].linear += solver_vels_inc[i].linear;
        solver_vels[i].angular += solver_vels_inc[i].angular;
    }
}

/// Apply warmstart impulses without relying on graph coloring.
/// This operates "gather-style" by iterating on constraints per-body.

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn warmstart_without_colors(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;

    for (var body_id = invocation_id.x; body_id < num_colliders; body_id += num_threads) {
        var first_constraint_id = 0u;
        let last_constraint_id = body_constraint_counts[body_id];
        if body_id != 0u {
            first_constraint_id = body_constraint_counts[body_id - 1u];
        }

        var solver_vel = solver_vels[body_id];

        for (var i = first_constraint_id; i < last_constraint_id; i += 1u) {
            let cid = body_constraint_ids[i];
            let solver_body_1 = constraints[cid].solver_body_a;
            let dir_a = constraints[cid].dir_a;
            let im_a = constraints[cid].im_a;
            let im_b = constraints[cid].im_b;
            let len = constraints[cid].len;

#if DIM == 3
            let tangent_a = constraints[cid].tangent_a;
#else
            let tangent_a = vec2(-dir_a.y, dir_a.x);
#endif

            for (var k = 0u; k < len; k++) {
                // Warmstart the normal part of the constraint.
                {
                    let c = constraints[cid].elements[k].normal_part;
                    if solver_body_1 == body_id {
                        solver_vel.linear += dir_a * im_a * c.impulse;
                        solver_vel.angular += c.ii_torque_dir_a * c.impulse;
                    } else {
                        solver_vel.linear += dir_a * im_b * -c.impulse_jacobi;
                        solver_vel.angular += c.ii_torque_dir_b * c.impulse_jacobi;
                    }
                }

                // Warmstart the tangent parts of the constraint.
                {
                    var c = constraints[cid].elements[k].tangent_part;

            #if DIM == 2
                    if solver_body_1 == body_id {
                        solver_vel.linear += tangent_a * im_a * c.impulse[0];
                        solver_vel.angular += c.ii_torque_dir_a[0] * c.impulse[0];
                    } else {
                        solver_vel.linear += tangent_a * im_b * -c.impulse_jacobi[0];
                        solver_vel.angular += c.ii_torque_dir_b[0] * c.impulse_jacobi[0];
                    }
            #else
                    let tangents_a = array(tangent_a, cross(dir_a, tangent_a));
                    if solver_body_1 == body_id {
                        solver_vel.linear +=
                            (tangents_a[0] * c.impulse[0] + tangents_a[1] * c.impulse[1]) * im_a;
                        solver_vel.angular +=
                            c.ii_torque_dir_a[0] * c.impulse[0] + c.ii_torque_dir_a[1] * c.impulse[1];
                    } else {
                        solver_vel.linear +=
                            (tangents_a[0] * -c.impulse[0] + tangents_a[1] * -c.impulse_jacobi[1]) * im_b;
                        solver_vel.angular +=
                            c.ii_torque_dir_b[0] * c.impulse[0] + c.ii_torque_dir_b[1] * c.impulse_jacobi[1];
                    }
            #endif
                }
            }
        }

        solver_vels[body_id] = solver_vel;
    }
}


@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn warmstart(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;

    for (var i = invocation_id.x; i < contacts_len; i += num_threads) {
        if constraints_colors[i] != curr_color {
            continue;
        }

        let solver_id1 = constraints[i].solver_body_a;
        let solver_id2 = constraints[i].solver_body_b;
        let dir_a = constraints[i].dir_a;
        let im_a = constraints[i].im_a;
        let im_b = constraints[i].im_b;

#if DIM == 3
        let tangent_a = constraints[i].tangent_a;
#else
        let tangent_a = vec2(-dir_a.y, dir_a.x);
#endif

        var solver_vel1 = solver_vels[solver_id1];
        var solver_vel2 = solver_vels[solver_id2];

        // TODO: unroll?
        for (var k = 0u; k < constraints[i].len; k++) {
            // Warmstart the normal part of the constraint.
            {
                let c = constraints[i].elements[k].normal_part;
                solver_vel1.linear += dir_a * im_a * c.impulse;
                solver_vel1.angular += c.ii_torque_dir_a * c.impulse;
                solver_vel2.linear += dir_a * im_b * -c.impulse;
                solver_vel2.angular += c.ii_torque_dir_b * c.impulse;
            }

            // Warmstart the tangent parts of the constraint.
            {
                var c = constraints[i].elements[k].tangent_part;

    #if DIM == 2
                solver_vel1.linear += tangent_a * im_a * c.impulse[0];
                solver_vel1.angular += c.ii_torque_dir_a[0] * c.impulse[0];
                solver_vel2.linear += tangent_a * im_b * -c.impulse[0];
                solver_vel2.angular += c.ii_torque_dir_b[0] * c.impulse[0];
    #else
                let tangents_a = array(tangent_a, cross(dir_a, tangent_a));
                solver_vel1.linear +=
                    (tangents_a[0] * c.impulse[0] + tangents_a[1] * c.impulse[1]) * im_a;
                solver_vel1.angular +=
                    c.ii_torque_dir_a[0] * c.impulse[0] + c.ii_torque_dir_a[1] * c.impulse[1];
                solver_vel2.linear +=
                    (tangents_a[0] * -c.impulse[0] + tangents_a[1] * -c.impulse[1]) * im_b;
                solver_vel2.angular +=
                    c.ii_torque_dir_b[0] * c.impulse[0] + c.ii_torque_dir_b[1] * c.impulse[1];
    #endif
            }
        }

        solver_vels[solver_id1] = solver_vel1;
        solver_vels[solver_id2] = solver_vel2;
    }
}

/// Main constraint solver iteration kernel (Projected Gauss-Seidel).
///
/// This is the core of the physics solver. It iteratively solves constraints by:
/// 1. Computing constraint violations (velocity errors)
/// 2. Calculating corrective impulses
/// 3. Applying impulses to update body velocities
/// 4. Projecting impulses to valid ranges
///
/// Parallelization:
/// Uses graph coloring to allow parallel solving. Only processes constraints
/// matching curr_color, ensuring no two threads modify the same body.
///
/// Algorithm per constraint:
/// For each contact point:
///   - Solve normal constraint (non-penetration):
///       dvel = J * v + rhs  (compute velocity error)
///       impulse = clamp(impulse - r * dvel, 0, ∞)  (compute corrective impulse)
///       v += J^T * impulse  (apply impulse to velocities)
///
///   - Solve tangent constraints (friction):
///       Similar to normal, but clamped to friction cone: |f_t| <= μ * f_n
///
/// @param invocation_id: Global thread ID
/// @param num_workgroups: Number of workgroups dispatched
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn step_gauss_seidel(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;

    // Grid-stride loop over all constraints
    for (var i = invocation_id.x; i < contacts_len; i += num_threads) {
        // Only process constraints of the current color (for parallelization)
        if constraints_colors[i] != curr_color {
            continue;
        }

        let solver_id1 = constraints[i].solver_body_a;
        let solver_id2 = constraints[i].solver_body_b;
        let dir_a = constraints[i].dir_a;
        let friction_coeff = constraints[i].limit;
        let im_a = constraints[i].im_a;
        let im_b = constraints[i].im_b;
        let cfm_factor = constraints[i].cfm_factor;

#if DIM == 3
        let tangent_a = constraints[i].tangent_a;
#else
        let tangent_a = vec2(-dir_a.y, dir_a.x);
#endif

        var solver_vel1 = solver_vels[solver_id1];
        var solver_vel2 = solver_vels[solver_id2];

        // TODO: unroll?
        for (var k = 0u; k < constraints[i].len; k++) {
            var limit = 0.0; // Friction impulse limit.

            // Solve the normal part of the constraint.
            {
                let c = constraints[i].elements[k].normal_part;

                let dvel = dot(dir_a, solver_vel1.linear) + gdot(c.torque_dir_a, solver_vel1.angular)
                    - dot(dir_a, solver_vel2.linear)
                    + gdot(c.torque_dir_b, solver_vel2.angular)
                    + c.rhs;
                let new_impulse = cfm_factor * max(c.impulse - c.r * dvel, 0.0);
                let delta_impulse = new_impulse - c.impulse;

                constraints[i].elements[k].normal_part.impulse = new_impulse;

                solver_vel1.linear += dir_a * im_a * delta_impulse;
                solver_vel1.angular += c.ii_torque_dir_a * delta_impulse;

                solver_vel2.linear += dir_a * im_b * -delta_impulse;
                solver_vel2.angular += c.ii_torque_dir_b * delta_impulse;
                limit = new_impulse * friction_coeff;
            }

            // Solve the tangent parts of the constraint.
            {
                var c = constraints[i].elements[k].tangent_part;

    #if DIM == 2
                let dvel = dot(tangent_a, solver_vel1.linear) + gdot(c.torque_dir_a[0], solver_vel1.angular)
                    - dot(tangent_a, solver_vel2.linear)
                    + gdot(c.torque_dir_b[0], solver_vel2.angular)
                    + c.rhs[0];
                let new_impulse = cfm_factor * clamp(c.impulse[0] - c.r[0] * dvel, -limit, limit);
                let delta_impulse = new_impulse - c.impulse[0];

                constraints[i].elements[k].tangent_part.impulse[0] = new_impulse;

                solver_vel1.linear += tangent_a * im_a * delta_impulse;
                solver_vel1.angular += c.ii_torque_dir_a[0] * delta_impulse;

                solver_vel2.linear += tangent_a * im_b * -delta_impulse;
                solver_vel2.angular += c.ii_torque_dir_b[0] * delta_impulse;
    #else
                let tangents_a = array(tangent_a, cross(dir_a, tangent_a));
                let dvel_0 = dot(tangents_a[0], solver_vel1.linear)
                    + gdot(c.torque_dir_a[0], solver_vel1.angular)
                    - dot(tangents_a[0], solver_vel2.linear)
                    + gdot(c.torque_dir_b[0], solver_vel2.angular)
                    + c.rhs[0];
                let dvel_1 = dot(tangents_a[1], solver_vel1.linear)
                    + gdot(c.torque_dir_a[1], solver_vel1.angular)
                    - dot(tangents_a[1], solver_vel2.linear)
                    + gdot(c.torque_dir_b[1], solver_vel2.angular)
                    + c.rhs[1];

                let dvel_00 = dvel_0 * dvel_0;
                let dvel_11 = dvel_1 * dvel_1;
                let dvel_01 = dvel_0 * dvel_1;
                let inv_lhs = (dvel_00 + dvel_11)
                    * maybe_inv(
                        dvel_00 * c.r[0] + dvel_11 * c.r[1] + dvel_01 * c.r[2],
                    );
                let delta_impulse = vec2(inv_lhs * dvel_0, inv_lhs * dvel_1);
                var new_impulse = c.impulse - delta_impulse;
                new_impulse = cap_magnitude(new_impulse, limit);

                let dlambda = new_impulse - c.impulse;
                constraints[i].elements[k].tangent_part.impulse = new_impulse;

                solver_vel1.linear +=
                    (tangents_a[0] * dlambda[0] + tangents_a[1] * dlambda[1]) * im_a;
                solver_vel1.angular +=
                    c.ii_torque_dir_a[0] * dlambda[0] + c.ii_torque_dir_a[1] * dlambda[1];

                solver_vel2.linear +=
                    (tangents_a[0] * -dlambda[0] + tangents_a[1] * -dlambda[1]) * im_b;
                solver_vel2.angular +=
                    c.ii_torque_dir_b[0] * dlambda[0] + c.ii_torque_dir_b[1] * dlambda[1];
    #endif
            }
        }

        solver_vels[solver_id1] = solver_vel1;
        solver_vels[solver_id2] = solver_vel2;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn integrate(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;

    if i < num_colliders {
        let vels = Body::Velocity(solver_vels[i].linear, solver_vels[i].angular);
        poses[i] = Body::integrateVelocity(poses[i], vels, local_mprops[i].com, params.dt);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn finalize(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;

    if i < num_colliders {
        vels[i].linear = solver_vels[i].linear;
        vels[i].angular = solver_vels[i].angular;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn remove_cfm_and_bias(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    if i < contacts_len {
        constraints[i].elements[0].normal_part.rhs = constraints[i].elements[0].normal_part.rhs_wo_bias;
        constraints[i].elements[1].normal_part.rhs = constraints[i].elements[1].normal_part.rhs_wo_bias;
#if DIM == 3
        constraints[i].elements[2].normal_part.rhs = constraints[i].elements[2].normal_part.rhs_wo_bias;
        constraints[i].elements[3].normal_part.rhs = constraints[i].elements[3].normal_part.rhs_wo_bias;
#endif
        constraints[i].cfm_factor = 1.0;
    }
}

fn contact_to_constraint(out_id: u32, indexed_contact: Contact::IndexedManifold) {
    var constraint = Constraint::TwoBodyConstraint();
    let id1 = indexed_contact.colliders.x;
    let id2 = indexed_contact.colliders.y;
    let contact = indexed_contact.contact;

    let mprops1 = mprops[id1];
    let mprops2 = mprops[id2];
    let pose1 = poses[id1];
    let pose2 = poses[id2];
    let vel1 = vels[id1];
    let vel2 = vels[id2];

    let force_dir1 = -Pose::mulVec(pose1, contact.normal_a);

    let cfm_factor = Params::contact_cfm_factor(params);
    let inv_dt = Params::inv_dt(params);
    let erp_inv_dt = Params::contact_erp_inv_dt(params);
    let allowed_linear_error = Params::allowed_linear_error(params);
    let max_corrective_velocity = Params::max_corrective_velocity(params);

    let friction = 0.5; // TODO(read from material properties)
    let restitution = 0.0; // TODO(deduce from material properties)

    var tangents1 = compute_tangent_contact_directions(force_dir1, vel1.linear, vel2.linear);
    constraint.dir_a = force_dir1;
    constraint.im_a = mprops1.inv_mass;
    constraint.im_b = mprops2.inv_mass;
    constraint.cfm_factor = cfm_factor;
    constraint.limit = friction;
    constraint.solver_body_a = id1;
    constraint.solver_body_b = id2;

#if DIM == 3
    constraint.tangent_a = tangents1[0];
#endif

    for (var k = 0u; k < contact.len; k++) {
        let pt = Pose::mulPt(pose1, contact.points_a[k].pt + contact.normal_a * contact.points_a[k].dist / 2.0);
        let dp1 = pt - mprops1.com;
        let dp2 = pt - mprops2.com;
        let contact_vel1 = vel1.linear + gcross_(vel1.angular, dp1);
        let contact_vel2 = vel2.linear + gcross_(vel2.angular, dp2);

        //
        // Normal part:
        //
        let torque_dir1 = gcross(dp1, force_dir1);
        let torque_dir2 = gcross(dp2, -force_dir1);
        let ii_torque_dir1 = mprops1.inv_inertia * torque_dir1;
        let ii_torque_dir2 = mprops2.inv_inertia * torque_dir2;
        let imsum = mprops1.inv_mass + mprops2.inv_mass;
        let projected_mass = inv(
            dot(force_dir1, imsum * force_dir1) +
            gdot(ii_torque_dir1, torque_dir1) +
            gdot(ii_torque_dir2, torque_dir2)
        );

        // TODO: handle is_bouncy?
        let dist = contact.points_a[k].dist;
        let normal_rhs_wo_bias = restitution * dot(contact_vel1 - contact_vel2, force_dir1)
            + max(dist, 0.0) * inv_dt;

        let rhs_bias = clamp(erp_inv_dt * (dist + allowed_linear_error),
            -max_corrective_velocity, 0.0);

        constraint.elements[k].normal_part = Constraint::TwoBodyConstraintNormalPart(
            torque_dir1,
            ii_torque_dir1,
            torque_dir2,
            ii_torque_dir2,
            normal_rhs_wo_bias, // := rhs
            normal_rhs_wo_bias, // := rhs_wo_bias
            0.0, // := impulse
            0.0, // := impulse_jacobi
            projected_mass, // := r
        );

        //
        // Tangent part:
        //
        for (var j = 0u; j < Constraint::SUB_LEN; j++) {
            let torque_dir1 = gcross(dp1, tangents1[j]);
            let torque_dir2 = gcross(dp2, -tangents1[j]);
            let ii_torque_dir1 = mprops1.inv_inertia * torque_dir1;
            let ii_torque_dir2 = mprops2.inv_inertia * torque_dir2;
            let r = dot(tangents1[j], imsum * tangents1[j])
                + gdot(ii_torque_dir1, torque_dir1)
                + gdot(ii_torque_dir2, torque_dir2);
//            let tangent_velocity = vec3(0.0);
//            let rhs_wo_bias = dot(tangent_velocity, tangents1[j]);
            let rhs_wo_bias = 0.0;

            constraint.elements[k].tangent_part.torque_dir_a[j] = torque_dir1;
            constraint.elements[k].tangent_part.ii_torque_dir_a[j] = ii_torque_dir1;
            constraint.elements[k].tangent_part.torque_dir_b[j] = torque_dir2;
            constraint.elements[k].tangent_part.ii_torque_dir_b[j] = ii_torque_dir2;
            constraint.elements[k].tangent_part.rhs[j] = rhs_wo_bias;
            constraint.elements[k].tangent_part.rhs_wo_bias[j] = rhs_wo_bias;
    #if DIM == 2
            constraint.elements[k].tangent_part.r[j] = inv(r);
    #else
            constraint.elements[k].tangent_part.r[j] = r;
    #endif
        }

        // NOTE: warmstart values are handled in a separate kernel.
#if DIM == 2
        constraint.elements[k].tangent_part.impulse = array(0.0);
        constraint.elements[k].tangent_part.impulse_jacobi = array(0.0);
#else
        constraint.elements[k].tangent_part.impulse = vec2(0.0, 0.0);
        constraint.elements[k].tangent_part.impulse_jacobi = vec2(0.0, 0.0);
#endif

    #if DIM == 3
        constraint.elements[k].tangent_part.r[2] = 2.0
            * (dot(constraint.elements[k].tangent_part.torque_dir_a[0], constraint.elements[k].tangent_part.ii_torque_dir_a[1])
                + dot(constraint.elements[k].tangent_part.torque_dir_b[0], constraint.elements[k].tangent_part.ii_torque_dir_b[1]));
    #endif

        // Builder.
        constraint_builders[out_id].infos[k].local_pt_a = Pose::invMulPt(pose1, pt);
        constraint_builders[out_id].infos[k].local_pt_b = Pose::invMulPt(pose2, pt);
//        constraint_builders[out_id].infos[k].tangent_vel = solver_contact.tangent_velocity; // TODO (conveyor belts)
        constraint_builders[out_id].infos[k].dist = dist;
        constraint_builders[out_id].infos[k].normal_vel = normal_rhs_wo_bias;
    }

    constraint.len = contact.len;
    constraints[out_id] = constraint;
}

fn inv(x: f32) -> f32 {
    return select(1.0 / x, 0.0, x == 0.0);
}


#if DIM == 2
fn orthonormal_vector(vec: vec2<f32>) -> vec2<f32> {
    return vec2(-vec.y, vec.x);
}

// TODO: share this from another module (it’s called perp in other modules).
fn gcross(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

fn gcross_(a: f32, b: vec2<f32>) -> vec2<f32> {
    return a * b;
}

fn gdot(a: f32, b: f32) -> f32 {
    return a * b;
}

fn compute_tangent_contact_directions(
    force_dir1: Vector,
    _linvel1: Vector,
    _linvel2: Vector,
) -> array<Vector, Constraint::SUB_LEN>
{
    return array(orthonormal_vector(force_dir1));
}
#else
fn orthonormal_vector(vec: vec3<f32>) -> vec3<f32> {
    let sign = select(sign(vec.z), 1.0, vec.z == 0.0);
    let a = -1.0 / (sign + vec.z);
    let b = vec.x * vec.y * a;
    return vec3(b, sign + vec.y * vec.y * a, -vec.y);
}

fn gcross(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return cross(a, b);
}

fn gcross_(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return cross(a, b);
}

fn gdot(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return dot(a, b);
}

fn compute_tangent_contact_directions(
    force_dir1: Vector,
    linvel1: Vector,
    linvel2: Vector,
) -> array<Vector, Constraint::SUB_LEN>
{
    // Compute the tangent direction. Pick the direction of
    // the linear relative velocity, if it is not too small.
    // Otherwise use a fallback direction.
    let relative_linvel = linvel1 - linvel2;
    var tangent_relative_linvel =
        relative_linvel - force_dir1 * dot(force_dir1, relative_linvel);

    let tangent_linvel_norm = length(tangent_relative_linvel);
    tangent_relative_linvel /= tangent_linvel_norm;

    const THRESHOLD: f32 = 1.0e-4;
    let use_fallback = tangent_linvel_norm < THRESHOLD;
    let tangent_fallback = orthonormal_vector(force_dir1);

    let tangent1 = select(tangent_relative_linvel, tangent_fallback, use_fallback);
    let bitangent1 = cross(force_dir1, tangent1);

    return array(tangent1, bitangent1);
}
#endif

fn maybe_inv(a: f32) -> f32 {
    const INV_EPSILON: f32 = 1.0e-20;
    return select(0.0, 1.0 / a, a < -INV_EPSILON || a > INV_EPSILON);
}

fn cap_magnitude(v: vec2<f32>, limit: f32) -> vec2<f32> {
    let n = length(v);
    return select(v, v * (limit / n), n > limit);
}
