#define_import_path wgrapier::body

#if DIM == 2
    #import wgebra::sim2 as Pose
    #import wgebra::rot2 as Rot
#else
    #import wgebra::sim3 as Pose
    #import wgebra::quat as Rot
#endif


/// The mass-properties of a rigid-body.
struct LocalMassProperties {
    // TODO: a representation with Quaternion & vec3 (for frame & principal inertia) would be much more compact and make
    //       this struct have the size of a mat4x4
#if DIM == 2
   /// The rigid-body’s inverse inertia tensor.
   inv_inertia_sqrt: f32,
#else
   inertia_ref_frame: Rot::Quat,
   inv_principal_inertia_sqrt: vec3<f32>,
#endif
   /// The rigid-body’s inverse mass along each coordinate axis.
   ///
   /// Allowing different values along each axis allows the user to specify 0 along each axis.
   /// By setting zero, the linear motion along the corresponding world-space axis will be locked.
   inv_mass: Vector,
   /// The rigid-body’s center of mass.
   com: Vector,
}

/// The mass-properties of a rigid-body.
struct WorldMassProperties {
    // TODO: a representation with Quaternion & vec3 (for frame & principal inertia) would be much more compact and make
    //       this struct have the size of a mat4x4
#if DIM == 2
   /// The rigid-body’s inverse inertia tensor.
   inv_inertia_sqrt: f32,
#else
   inv_inertia_sqrt: mat3x3<f32>,
#endif
   /// The rigid-body’s inverse mass along each coordinate axis.
   ///
   /// Allowing different values along each axis allows the user to specify 0 along each axis.
   /// By setting zero, the linear motion along the corresponding world-space axis will be locked.
   inv_mass: Vector,
   /// The rigid-body’s center of mass.
   com: Vector,
}

/// An impulse (linear and angular/torque)
struct Impulse {
    /// A linear impulse.
    linear: Vector,
    /// An angular impulse (torque impulse).
    angular: AngVector,
}

/// A force and torque.
struct Force {
    /// A linear force.
    linear: Vector,
    /// An angular force (torque).
    angular: AngVector,
}

/// A linear and angular velocity.
struct Velocity {
    /// The linear (translational) part of the velocity.
    linear: Vector,
    /// The angular (rotational) part of the velocity.
    angular: AngVector,
}

/// A rigid-body pose and its velocity.
struct RigidBodyState {
    /// The rigid-body’s pose (translation, rotation, uniform scale).
    pose: Transform,
    /// The rigid-body’s velocity (translational and rotational).
    velocity: Velocity,
}

/// Computes new velocities after applying the given impulse.
fn applyImpulse(mprops: WorldMassProperties, velocity: Velocity, imp: Impulse) -> Velocity {
    let acc_lin = mprops.inv_mass * imp.linear;
    let acc_ang = mprops.inv_inertia_sqrt * (mprops.inv_inertia_sqrt * imp.angular);
    return Velocity(velocity.linear + acc_lin, velocity.angular + acc_ang);
}


/// Computes new velocities after integrating forces by a timestep equal to `dt`.
fn integrateForces(mprops: WorldMassProperties, velocity: Velocity, force: Force, dt: f32) -> Velocity {
    let acc_lin = mprops.inv_mass * force.linear;
    let acc_ang = mprops.inv_inertia_sqrt * (mprops.inv_inertia_sqrt * force.angular);
    return Velocity(velocity.linear + acc_lin * dt, velocity.angular + acc_ang * dt);
}

#if DIM == 2
/// Computes a new pose after integrating velocitie by a timestep equal to `dt`.
fn integrateVelocity(pose: Transform, vels: Velocity, local_com: Vector, dt: f32) -> Transform {
    let init_com = Pose::mulPt(pose, local_com);
    let init_tra = pose.translation;
    let init_scale = pose.scale;

    let delta_ang = Rot::fromAngle(vels.angular * dt);
    let delta_lin = vels.linear * dt;

    let new_translation =
        init_com + Rot::mulVec(delta_ang, (init_tra - init_com)) * init_scale + delta_lin;
    let new_rotation = Rot::mul(delta_ang, pose.rotation);

    return Transform(new_rotation, new_translation, init_scale);
}

/// Computes the new world-space mass-properties based on the local-space mass-properties and its transform.
fn updateMprops(pose: Transform, local_mprops: LocalMassProperties) -> WorldMassProperties {
    let world_com = Pose::mulPt(pose, local_mprops.com);
    return WorldMassProperties(local_mprops.inv_inertia_sqrt, local_mprops.inv_mass, world_com);
}

/// Computes the linear velocity at a given point.
fn velocity_at_point(center_of_mass: Vector, vels: Velocity, point: Vector) -> Vector {
    let lever_arm = point - center_of_mass;
    return vels.linear + vels.angular * vec2(-lever_arm.y, lever_arm.x);
}
#else
/// Computes a new pose after integrating velocitie by a timestep equal to `dt`.
fn integrateVelocity(pose: Transform, vels: Velocity, local_com: Vector, dt: f32) -> Transform {
    let init_com = Pose::mulPt(pose, local_com);
    let init_tra = pose.translation_scale.xyz;
    let init_scale = pose.translation_scale.w;

    let delta_ang = Rot::fromScaledAxis(vels.angular * dt);
    let delta_lin = vels.linear * dt;

    let new_translation =
        init_com + Rot::mulVec(delta_ang, (init_tra - init_com)) * init_scale + delta_lin;
    let new_rotation = Rot::renormalizeFast(Rot::mul(delta_ang, pose.rotation));

    return Transform(new_rotation, vec4(new_translation, init_scale));
}

/// Computes the new world-space mass-properties based on the local-space mass-properties and its transform.
fn updateMprops(pose: Transform, local_mprops: LocalMassProperties) -> WorldMassProperties {
    let world_com = Pose::mulPt(pose, local_mprops.com);
    let rot_mat = Rot::toMatrix(Rot::mul(pose.rotation, local_mprops.inertia_ref_frame));
    let diag = mat3x3(
        vec3(local_mprops.inv_principal_inertia_sqrt.x, 0.0, 0.0),
        vec3(0.0, local_mprops.inv_principal_inertia_sqrt.y, 0.0),
        vec3(0.0, 0.0, local_mprops.inv_principal_inertia_sqrt.z),
    );
    let world_inv_inertia = rot_mat * diag * transpose(rot_mat);

    return WorldMassProperties(world_inv_inertia, local_mprops.inv_mass, world_com);
}

/// Computes the linear velocity at a given point.
fn velocity_at_point(com: Vector, vels: Velocity, point: Vector) -> Vector {
    return vels.linear + cross(vels.angular, point - com);
}
#endif