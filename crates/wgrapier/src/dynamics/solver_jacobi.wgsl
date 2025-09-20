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
var<storage, read_write> contacts: array<Contact::IndexedContact>;
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

@group(1) @binding(0)
var<uniform> params: Params::SimParams;
@group(1) @binding(1)
var<storage, read> poses: array<Transform>;
@group(1) @binding(2)
var<storage, read_write> vels: array<Body::Velocity>;
@group(1) @binding(3)
var<storage, read> mprops: array<Body::WorldMassProperties>;
@group(1) @binding(4)
var<uniform> num_colliders: u32;

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn init_constraints(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < contacts_len; i += num_threads) {
        constraints[i] = contact_to_constraint(contacts[i]);

        let body1 = contacts[i].colliders.x;
        let body2 = contacts[i].colliders.y;
        atomicAdd(&body_constraint_counts_atomic[body1], 1u);
        atomicAdd(&body_constraint_counts_atomic[body2], 1u);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn sort_constraints(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < contacts_len; i += num_threads) {
        let body1 = contacts[i].colliders.x;
        let body2 = contacts[i].colliders.y;
        let id1 = atomicAdd(&body_constraint_counts_atomic[body1], 1u);
        let id2 = atomicAdd(&body_constraint_counts_atomic[body2], 1u);
        body_constraint_ids[id1] = i;
        body_constraint_ids[id2] = i;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn cleanup(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < num_colliders; i += num_threads) {
        body_constraint_counts[i] = 0;

        // HACK: to handle static bodies.
        if any(mprops[i].inv_mass != vec3(0.0f, 0.0, 0.0)) {
            solver_vels[i].linear = vels[i].linear;
            solver_vels[i].angular = Inv::inv3(mprops[i].inv_inertia_sqrt) * vels[i].angular;
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
            let solver_id1 = constraints[cid].solver_vel_a;
            let solver_id2 = constraints[cid].solver_vel_b;
            let dir_a = constraints[cid].dir_a;
            let im_a = constraints[cid].im_a;
            let im_b = constraints[cid].im_b;
            let cfm_factor = constraints[cid].cfm_factor;

            let c = constraints[cid].elements.normal_part;

            let solver_vel1 = solver_vels[solver_id1];
            let solver_vel2 = solver_vels[solver_id2];
            var impulse = c.impulse_a;

            if solver_id1 != body_id {
                impulse = c.impulse_b;
            }

            // Solve the normal part of the constraint.
            let dvel = dot(dir_a, solver_vel1.linear) + dot(c.gcross_a, solver_vel1.angular)
                - dot(dir_a, solver_vel2.linear)
                + dot(c.gcross_b, solver_vel2.angular)
                + c.rhs;
            let new_impulse = cfm_factor * max(impulse - c.r * dvel, 0.0);
            let delta_impulse = new_impulse - impulse;

            if solver_id1 == body_id {
                constraints[cid].elements.normal_part.impulse_a = new_impulse;
                solver_vel.linear += dir_a * im_a * delta_impulse;
                solver_vel.angular += c.gcross_a * delta_impulse;
            } else {
                constraints[cid].elements.normal_part.impulse_b = new_impulse;
                solver_vel.linear += dir_a * im_b * -delta_impulse;
                solver_vel.angular += c.gcross_b * delta_impulse;
            }

            // TODO: also solve the tangent part of the constraint.
        }

        solver_vels_out[body_id] = solver_vel;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn step_gauss_seidel(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if invocation_id.x != 0 {
        return;
    }

    for (var i = 0u; i < contacts_len; i += 1u) {
        let solver_id1 = constraints[i].solver_vel_a;
        let solver_id2 = constraints[i].solver_vel_b;
        let dir_a = constraints[i].dir_a;
        let im_a = constraints[i].im_a;
        let im_b = constraints[i].im_b;
        let cfm_factor = constraints[i].cfm_factor;

        var solver_vel1 = solver_vels[solver_id1];
        var solver_vel2 = solver_vels[solver_id2];

        // Solve the normal part of the constraint.
        let c = constraints[i].elements.normal_part;

        let dvel = dot(dir_a, solver_vel1.linear) + dot(c.gcross_a, solver_vel1.angular)
            - dot(dir_a, solver_vel2.linear)
            + dot(c.gcross_b, solver_vel2.angular)
            + c.rhs;
        let new_impulse = cfm_factor * max(c.impulse_a - c.r * dvel, 0.0);
        let delta_impulse = new_impulse - c.impulse_a;

        constraints[i].elements.normal_part.impulse_a = new_impulse;

        solver_vel1.linear += dir_a * im_a * delta_impulse;
        solver_vel1.angular += c.gcross_a * delta_impulse;

        solver_vel2.linear += dir_a * im_b * -delta_impulse;
        solver_vel2.angular += c.gcross_b * delta_impulse;

        // TODO: also solve the tangent part of the constraint.

        solver_vels[solver_id1] = solver_vel1;
        solver_vels[solver_id2] = solver_vel2;
    }
}



@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn finalize(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;

    if i < num_colliders {
        vels[i].linear = solver_vels[i].linear;
        vels[i].angular = mprops[i].inv_inertia_sqrt * solver_vels[i].angular;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn remove_cfm_and_bias(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    if i < contacts_len {
        constraints[i].elements.normal_part.rhs = constraints[i].elements.normal_part.rhs_wo_bias;
        constraints[i].cfm_factor = 1.0;
    }
}

fn contact_to_constraint(indexed_contact: Contact::IndexedContact) -> Constraint::TwoBodyConstraint {
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

    let friction = 0.7; // TODO(read from material properties)
    let restitution = 0.0; // TODO(deduce from material properties)
    let warmstart_impulse = 0.0; // TODO(for warmstart support)

    let tangents1 = compute_tangent_contact_directions(force_dir1, vel1.linear, vel2.linear);
    constraint.dir_a = force_dir1;
    constraint.tangent_a = tangents1[0];
    constraint.im_a = mprops1.inv_mass;
    constraint.im_b = mprops2.inv_mass;
    constraint.cfm_factor = cfm_factor;
    constraint.limit = friction;
    constraint.solver_vel_a = id1;
    constraint.solver_vel_b = id2;


    let pt = (Pose::mulPt(pose1, contact.point_a) + Pose::mulPt(pose2, contact.point_b)) / 2.0;
    let dp1 = pt - mprops1.com;
    let dp2 = pt - mprops2.com;
    let contact_vel1 = vel1.linear + cross(vel1.angular, dp1);
    let contact_vel2 = vel2.linear + cross(vel2.angular, dp2);

    //
    // Normal part:
    //
    let gcross1 = mprops1.inv_inertia_sqrt * cross(dp1, force_dir1);
    let gcross2 = mprops2.inv_inertia_sqrt * cross(dp2, -force_dir1);
    let imsum = mprops1.inv_mass + mprops2.inv_mass;
    let projected_mass = inv(
        dot(force_dir1, imsum * force_dir1) +
        dot(gcross1, gcross1) +
        dot(gcross2, gcross2)
    );

    // TODO: handle is_bouncy?
    let dist = contact.dist;
    let normal_rhs_wo_bias = restitution * dot(contact_vel1 - contact_vel2, force_dir1)
        + max(dist, 0.0) * inv_dt;

    // NOTE: for TGS, the bias calculation should be separate.
    let rhs_bias = clamp(erp_inv_dt * (dist + allowed_linear_error),
        -max_corrective_velocity, 0.0);

    constraint.elements.normal_part = Constraint::TwoBodyConstraintNormalPart(
        gcross1,
        gcross2,
        normal_rhs_wo_bias + rhs_bias, // := rhs
        normal_rhs_wo_bias, // := rhs_wo_bias
        warmstart_impulse,
        0.0, // := impulse_accumulator
        projected_mass,
        // TODO(block solver)
        vec2(0.0, 0.0), // := r_mat_elts
    );

    //
    // Tangent part:
    //
    for (var j = 0; j < 2; j++) {
        let gcross1 = mprops1.inv_inertia_sqrt * cross(dp1, tangents1[j]);
        let gcross2 = mprops2.inv_inertia_sqrt * cross(dp2, -tangents1[j]);
        let r = dot(tangents1[j], imsum * tangents1[j])
            + dot(gcross1, gcross1)
            + dot(gcross2, gcross2);
        let tangent_velocity = vec3(0.0);
        let rhs_wo_bias = dot(tangent_velocity, tangents1[j]);

        constraint.elements.tangent_part.gcross_a[j] = gcross1;
        constraint.elements.tangent_part.gcross_b[j] = gcross2;
        constraint.elements.tangent_part.rhs[j] = rhs_wo_bias;
        constraint.elements.tangent_part.rhs_wo_bias[j] = rhs_wo_bias;
#if DIM == 2
        constraint.elements.tangent_part.r[j] = inv(r);
#else
        constraint.elements.tangent_part.r[j] = r;
#endif
    }

    constraint.elements.tangent_part.impulse = array(0.0, 0.0); // TODO(warmstart)
    constraint.elements.tangent_part.impulse_accumulator = array(0.0, 0.0);

#if DIM == 3
    constraint.elements.tangent_part.r[2] = 2.0
        * (dot(constraint.elements.tangent_part.gcross_a[0], constraint.elements.tangent_part.gcross_a[1])
            + dot(constraint.elements.tangent_part.gcross_b[0], constraint.elements.tangent_part.gcross_b[1]));
#endif

    return constraint;
}

fn inv(x: f32) -> f32 {
    return select(1.0 / x, 0.0, x == 0.0);
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

#if DIM == 2
fn orthonormal_vector(vec: vec2<f32>) -> vec2<f32> {
    return vec2(-vec.y, vec.x);
}
#else
fn orthonormal_vector(vec: vec3<f32>) -> vec3<f32> {
    let sign = select(sign(vec.z), 1.0, vec.z == 0.0);
    let a = -1.0 / (sign + vec.z);
    let b = vec.x * vec.y * a;
    return vec3(b, sign + vec.y * vec.y * a, -vec.y);
}
#endif