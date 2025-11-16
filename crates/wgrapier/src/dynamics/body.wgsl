//! Rigid-body dynamics data structures and integration
//!
//! This shader defines the fundamental data structures and operations for rigid-body
//! dynamics, including mass properties, forces, velocities, and integration routines.
//!
//! Key Concepts:
//! - Mass Properties: Defines how a body resists linear and angular motion
//! - Velocity Integration: Semi-implicit Euler integration for pose updates
//! - Impulse Application: Computing velocity changes from constraint forces
//!
//! Dimension Support:
//! - 2D: Rotation represented as scalar angle, inertia as scalar
//! - 3D: Rotation represented as quaternion, inertia as 3x3 matrix

#define_import_path wgrapier::body

#if DIM == 2
    #import wgebra::sim2 as Pose
    #import wgebra::rot2 as Rot
#else
    #import wgebra::sim3 as Pose
    #import wgebra::quat as Rot
#endif


/// The mass-properties of a rigid-body in local (body-space) coordinates.
///
/// Local mass properties are defined relative to the body's local coordinate frame
/// and remain constant unless the body's shape changes.
struct LocalMassProperties {
    // TODO: a representation with Quaternion & vec3 (for frame & principal inertia) would be much more compact and make
    //       this struct have the size of a mat4x4
#if DIM == 2
   /// The inverse inertia tensor (scalar in 2D).
   inv_inertia: f32,
#else
   /// The reference frame for the principal inertia axes (3D only).
   ///
   /// Defines the orientation of the principal axes of inertia in the body's local frame.
   /// This is combined with the body's world rotation to compute world-space inertia.
   inertia_ref_frame: Rot::Quat,

   /// The inverse principal inertia components (3D only).
   ///
   /// The three components correspond to the principal moments of inertia about the
   /// body's principal axes.
   inv_principal_inertia: vec3<f32>,
#endif
   /// The rigid-body's inverse mass along each coordinate axis.
   ///
   /// Inverse mass is used for efficient physics calculations (force = inv_mass * acceleration).
   /// Per-axis values allow locking motion along specific world-space axes by setting to zero.
   /// For unconstrained bodies, all components should be equal to 1/mass.
   inv_mass: Vector,

   /// The rigid-body's center of mass in local coordinates.
   ///
   /// All forces and torques are applied relative to this point. The center of mass
   /// position affects how rotational and translational motion are coupled.
   com: Vector,
}

/// The mass-properties of a rigid-body in world-space coordinates.
///
/// World mass properties are computed from local mass properties and the body's
/// current pose. They are updated each frame.
struct WorldMassProperties {
    // TODO: a representation with Quaternion & vec3 (for frame & principal inertia) would be much more compact and make
    //       this struct have the size of a mat4x4
#if DIM == 2
   /// The inverse inertia tensor in world space (scalar in 2D).
   ///
   /// In 2D, this is the same as the local-space value since rotation doesn't affect
   /// scalar inertia. Kept separate for API consistency with 3D.
   inv_inertia: f32,
#else
   /// The inverse inertia tensor in world space (3x3 matrix in 3D).
   ///
   /// Computed by rotating the local principal inertia into world space:
   /// inv_inertia_world = R * diag(inv_principal_sqrt) * R^T
   /// where R is the combined rotation of the body and principal axes frame.
   inv_inertia: mat3x3<f32>,
#endif
   /// The rigid-body's inverse mass along each coordinate axis in world space.
   ///
   /// This is typically the same as local inverse mass.
   inv_mass: Vector,

   /// The rigid-body's center of mass in world-space coordinates.
   ///
   /// Updated each frame by transforming the local center of mass by the body's pose.
   /// Used for computing lever arms in constraint solving.
   com: Vector,
}

/// An impulse (instantaneous change in momentum).
///
/// An impulse represents a sudden change in velocity, typically from collision response or constraint correction.
struct Impulse {
    /// Linear impulse component (change in linear momentum).
    ///
    /// Applied to change the body's linear velocity.
    linear: Vector,

    /// Angular impulse component (change in angular momentum / torque impulse).
    ///
    /// Applied to change the body's angular velocity.
    angular: AngVector,
}

/// A force and torque applied to a rigid body.
///
/// Forces are integrated over time to change velocities. Unlike impulses which
/// provide instantaneous velocity changes, forces survive accross timesteps.
struct Force {
    /// Linear force component.
    ///
    /// Causes linear acceleration. Units: Newtons (kg⋅m/s²)
    linear: Vector,

    /// Angular force component (torque).
    ///
    /// Causes angular acceleration. Units: N⋅m (2D: scalar, 3D: vector)
    angular: AngVector,
}

/// Linear and angular velocity of a rigid body.
///
/// Represents the instantaneous rate of change of position and orientation.
struct Velocity {
    /// Linear (translational) velocity.
    ///
    /// Rate of change of position. Units: m/s
    linear: Vector,

    /// Angular (rotational) velocity.
    ///
    /// Rate of change of orientation. In 2D: scalar angular velocity (rad/s).
    /// In 3D: angular velocity vector along rotation axis (rad/s).
    angular: AngVector,
}

/// Complete state of a rigid body (pose and velocity).
///
/// Combines kinematic state (pose) with dynamic state (velocity) for a rigid body.
struct RigidBodyState {
    /// The rigid-body's pose (position, orientation, and uniform scale).
    pose: Transform,

    /// The rigid-body's velocity (linear and angular).
    velocity: Velocity,
}

/// Applies an impulse to a rigid body, computing the resulting velocity change.
///
/// This function implements the fundamental impulse-velocity relationship:
/// Δv_linear = impulse_linear / mass
/// Δv_angular = I⁻¹ * impulse_angular (where I is the inertia tensor)
///
/// Parameters:
/// - mprops: The body's world-space mass properties
/// - velocity: The body's current velocity
/// - imp: The impulse to apply
///
/// Returns: The updated velocity after applying the impulse
fn applyImpulse(mprops: WorldMassProperties, velocity: Velocity, imp: Impulse) -> Velocity {
    // Linear velocity change: Δv = F⋅Δt / m = impulse * inv_mass
    let acc_lin = mprops.inv_mass * imp.linear;

    // Angular velocity change: Δω = I⁻¹ * τ⋅Δt = I⁻¹ * angular_impulse
    let acc_ang = mprops.inv_inertia * imp.angular;

    return Velocity(velocity.linear + acc_lin, velocity.angular + acc_ang);
}


/// Integrates forces over a timestep to compute velocity changes.
///
/// This implements explicit (forward) Euler integration for force application:
/// v_new = v_old + (F / m) * dt
/// ω_new = ω_old + (I⁻¹ * τ) * dt
///
/// Parameters:
/// - mprops: The body's world-space mass properties
/// - velocity: The body's current velocity
/// - force: The force to apply
/// - dt: The timestep duration (seconds)
///
/// Returns: The updated velocity after integrating the force
fn integrateForces(mprops: WorldMassProperties, velocity: Velocity, force: Force, dt: f32) -> Velocity {
    // Linear acceleration: a = F / m
    let acc_lin = mprops.inv_mass * force.linear;

    // Angular acceleration: α = I⁻¹ * τ
    let acc_ang = mprops.inv_inertia * force.angular;

    // Explicit Euler: v_new = v_old + a * dt
    return Velocity(velocity.linear + acc_lin * dt, velocity.angular + acc_ang * dt);
}

#if DIM == 2
/// Integrates velocity over a timestep to compute the new pose (2D version).
///
/// This implements semi-implicit Euler integration for rigid body motion, integrating
/// around the center of mass to properly couple rotational and translational motion.
///
/// Algorithm (2D):
/// 1. Compute world-space center of mass
/// 2. Apply angular velocity to compute rotation change: Δθ = ω * dt
/// 3. Apply linear velocity for translation: Δp = v * dt
/// 4. Rotate the body's offset (translation - COM) around COM
/// 5. Combine rotation and translation to get new pose
///
/// Parameters:
/// - pose: Current pose (rotation, translation, scale)
/// - vels: Current velocities (linear and angular)
/// - local_com: Center of mass in body-local coordinates
/// - dt: Timestep duration (seconds)
///
/// Returns: The new pose after integration
fn integrateVelocity(pose: Transform, vels: Velocity, local_com: Vector, dt: f32) -> Transform {
    // Transform local COM to world space
    let init_com = Pose::mulPt(pose, local_com);
    let init_tra = pose.translation;
    let init_scale = pose.scale;

    // Integrate angular velocity: θ_new = θ_old + ω * dt (in 2D, build rotation from angle)
    let delta_ang = Rot::fromAngle(vels.angular * dt);

    // Integrate linear velocity: p_new = p_old + v * dt
    let delta_lin = vels.linear * dt;

    // New translation: rotate the offset around COM, then add linear displacement
    // new_translation = COM + Rotate(translation - COM) + linear_displacement
    let new_translation =
        init_com + Rot::mulVec(delta_ang, (init_tra - init_com)) * init_scale + delta_lin;

    // Compose rotations: new_rotation = delta_rotation * old_rotation
    let new_rotation = Rot::mul(delta_ang, pose.rotation);

    return Transform(new_rotation, new_translation, init_scale);
}

/// Updates world-space mass properties from local properties and current pose (2D version).
///
/// Transforms mass properties from body-local space to world space. In 2D, only the
/// center of mass needs transformation since inertia is a scalar invariant under rotation.
///
/// Parameters:
/// - pose: Current pose of the body
/// - local_mprops: Mass properties in body-local coordinates
///
/// Returns: Mass properties in world-space coordinates
fn updateMprops(pose: Transform, local_mprops: LocalMassProperties) -> WorldMassProperties {
    // Transform COM from local to world space
    let world_com = Pose::mulPt(pose, local_mprops.com);

    // In 2D, inertia is scalar and doesn't change with rotation
    return WorldMassProperties(local_mprops.inv_inertia, local_mprops.inv_mass, world_com);
}

/// Computes the linear velocity at a specific point on a rigid body (2D version).
///
/// For a rigid body, the velocity at any point is composed of:
/// v_point = v_com + ω × r
/// where r is the vector from COM to the point.
///
/// In 2D, the cross product ω × r (scalar × vector) gives: ω * perpendicular(r)
///
/// Parameters:
/// - center_of_mass: The body's center of mass in world space
/// - vels: The body's velocities (linear and angular)
/// - point: The point at which to compute velocity
///
/// Returns: The linear velocity at the given point
fn velocity_at_point(center_of_mass: Vector, vels: Velocity, point: Vector) -> Vector {
    // Lever arm from COM to point
    let lever_arm = point - center_of_mass;

    // In 2D: ω × r = ω * perpendicular(r) = ω * (-r.y, r.x)
    return vels.linear + vels.angular * vec2(-lever_arm.y, lever_arm.x);
}
#else
/// Integrates velocity over a timestep to compute the new pose (3D version).
///
/// This implements semi-implicit Euler integration for rigid body motion in 3D, using
/// quaternions for rotation and integrating around the center of mass.
///
/// Parameters:
/// - pose: Current pose (quaternion rotation, translation+scale)
/// - vels: Current velocities (linear and angular velocity vector)
/// - local_com: Center of mass in body-local coordinates
/// - dt: Timestep duration (seconds)
///
/// Returns: The new pose after integration
fn integrateVelocity(pose: Transform, vels: Velocity, local_com: Vector, dt: f32) -> Transform {
    // Transform local COM to world space
    let init_com = Pose::mulPt(pose, local_com);
    let init_tra = pose.translation_scale.xyz;
    let init_scale = pose.translation_scale.w;

    // Integrate angular velocity using exponential map: q_delta = exp(ω * dt / 2)
    // fromScaledAxis computes the quaternion from the scaled rotation vector
    let delta_ang = Rot::fromScaledAxis(vels.angular * dt);

    // Integrate linear velocity: p_new = p_old + v * dt
    let delta_lin = vels.linear * dt;

    // New translation: rotate the offset around COM, then add linear displacement
    // new_translation = COM + Rotate(translation - COM) + linear_displacement
    let new_translation =
        init_com + Rot::mulVec(delta_ang, (init_tra - init_com)) * init_scale + delta_lin;

    // Compose quaternions and renormalize to prevent numerical drift
    // q_new = normalize(q_delta * q_old)
    let new_rotation = Rot::renormalizeFast(Rot::mul(delta_ang, pose.rotation));

    return Transform(new_rotation, vec4(new_translation, init_scale));
}

/// Updates world-space mass properties from local properties and current pose (3D version).
///
/// Transforms mass properties from body-local space to world space. In 3D, the inertia
/// tensor must be rotated into world space, which involves transforming the principal
/// inertia axes by the body's rotation.
///
/// Parameters:
/// - pose: Current pose of the body
/// - local_mprops: Mass properties in body-local coordinates
///
/// Returns: Mass properties in world-space coordinates
fn updateMprops(pose: Transform, local_mprops: LocalMassProperties) -> WorldMassProperties {
    // Transform COM from local to world space
    let world_com = Pose::mulPt(pose, local_mprops.com);

    // Combine body rotation with principal axes frame rotation
    let rot_mat = Rot::toMatrix(Rot::mul(pose.rotation, local_mprops.inertia_ref_frame));

    // Create diagonal matrix from principal inertia components
    let diag = mat3x3(
        vec3(local_mprops.inv_principal_inertia.x, 0.0, 0.0),
        vec3(0.0, local_mprops.inv_principal_inertia.y, 0.0),
        vec3(0.0, 0.0, local_mprops.inv_principal_inertia.z),
    );

    // Transform inertia to world space: I_world = R * I_principal * R^T
    let world_inv_inertia = rot_mat * diag * transpose(rot_mat);

    return WorldMassProperties(world_inv_inertia, local_mprops.inv_mass, world_com);
}

/// Computes the linear velocity at a specific point on a rigid body (3D version).
///
/// For a rigid body, the velocity at any point is composed of:
/// v_point = v_com + ω × r
/// where r is the vector from COM to the point, and × is the cross product.
///
/// Parameters:
/// - com: The body's center of mass in world space
/// - vels: The body's velocities (linear and angular velocity vector)
/// - point: The point at which to compute velocity
///
/// Returns: The linear velocity at the given point
fn velocity_at_point(com: Vector, vels: Velocity, point: Vector) -> Vector {
    // v_point = v_linear + ω × (point - COM)
    return vels.linear + cross(vels.angular, point - com);
}
#endif