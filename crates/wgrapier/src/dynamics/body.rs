//! Rigid-body definitions, mass properties, velocities, and GPU storage.
//!
//! This module provides the core data structures for representing rigid bodies on the GPU,
//! including their poses, velocities, forces, and mass properties. It also provides
//! [`GpuBodySet`] for managing collections of rigid bodies in GPU memory.

use encase::ShaderType;
use num_traits::Zero;
use rapier::geometry::ColliderHandle;
use rapier::prelude::MassProperties;
use rapier::{
    dynamics::{RigidBodyHandle, RigidBodySet},
    geometry::ColliderSet,
};
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgebra::{WgQuat, WgSim2, WgSim3};
use wgparry::math::{AngVector, AngularInertia, GpuSim, Point, Vector};
use wgparry::shapes::{GpuShape, ShapeBuffers};
use wgparry::{dim_shader_defs, substitute_aliases};
use wgpu::{BufferUsages, Device};

#[cfg(feature = "dim3")]
use nalgebra::Vector4;
#[cfg(feature = "dim3")]
use wgebra::GpuSim3;

#[derive(ShaderType, Copy, Clone, PartialEq)]
#[repr(C)]
/// Linear and angular forces with a layout compatible with the corresponding WGSL struct.
pub struct GpuForce {
    /// The linear part of the force.
    pub linear: Vector<f32>,
    /// The angular part of the force (aka. the torque).
    pub angular: AngVector<f32>,
}

#[derive(ShaderType, Copy, Clone, PartialEq, Default, Debug)]
#[repr(C)]
/// Linear and angular velocities with a layout compatible with the corresponding WGSL struct.
pub struct GpuVelocity {
    /// The linear (translational) velocity.
    pub linear: Vector<f32>,
    /// The angular (rotational) velocity.
    pub angular: AngVector<f32>,
}

#[derive(ShaderType, Copy, Clone, PartialEq)]
#[repr(C)]
/// Rigid-body mass-properties, with a layout compatible with the corresponding WGSL struct.
pub struct GpuLocalMassProperties {
    /// Square root of inverse principal inertia (scalar in 2D).
    #[cfg(feature = "dim2")]
    pub inv_principal_inertia: f32,
    #[cfg(feature = "dim3")]
    inv_ref_frame: Vector4<f32>,
    /// Square root of inverse principal inertia tensor (3D vector in 3D).
    #[cfg(feature = "dim3")]
    pub inv_principal_inertia: nalgebra::Vector3<f32>,
    /// The inverse mass.
    pub inv_mass: Vector<f32>,
    /// The center-of-mass.
    pub com: Vector<f32>, // ShaderType isn’t implemented for Point
}

#[derive(ShaderType, Copy, Clone, PartialEq)]
#[repr(C)]
/// Rigid-body mass-properties, with a layout compatible with the corresponding WGSL struct.
pub struct GpuWorldMassProperties {
    /// The inverse angular inertia tensor.
    pub inv_inertia: AngularInertia<f32>,
    /// The inverse mass.
    pub inv_mass: Vector<f32>,
    /// The center-of-mass.
    pub com: Vector<f32>, // ShaderType isn’t implemented for Point
}

impl From<MassProperties> for GpuLocalMassProperties {
    fn from(props: MassProperties) -> Self {
        GpuLocalMassProperties {
            inv_principal_inertia: props.inv_principal_inertia,
            #[cfg(feature = "dim3")]
            inv_ref_frame: props.principal_inertia_local_frame.coords,
            inv_mass: Vector::repeat(props.inv_mass),
            com: props.local_com.coords,
        }
    }
}

impl Default for GpuLocalMassProperties {
    fn default() -> Self {
        GpuLocalMassProperties {
            #[rustfmt::skip]
            #[cfg(feature = "dim2")]
            inv_principal_inertia: 1.0,
            #[cfg(feature = "dim3")]
            inv_ref_frame: Vector4::w(),
            #[cfg(feature = "dim3")]
            inv_principal_inertia: Vector::repeat(1.0),
            inv_mass: Vector::repeat(1.0),
            com: Vector::zeros(),
        }
    }
}

impl Default for GpuWorldMassProperties {
    fn default() -> Self {
        GpuWorldMassProperties {
            #[rustfmt::skip]
            #[cfg(feature = "dim2")]
            inv_inertia: 1.0,
            #[cfg(feature = "dim3")]
            inv_inertia: AngularInertia::identity(),
            inv_mass: Vector::repeat(1.0),
            com: Vector::zeros(),
        }
    }
}
/// A set of rigid-bodies stored on the gpu.
pub struct GpuBodySet {
    len: u32,
    shapes_data: Vec<GpuShape>, // TODO: exists only for convenience in the MPM simulation.
    pub(crate) mprops: GpuVector<GpuWorldMassProperties>,
    pub(crate) local_mprops: GpuVector<GpuLocalMassProperties>,
    pub(crate) vels: GpuVector<GpuVelocity>,
    pub(crate) poses: GpuVector<GpuSim>,
    // TODO: support other shape types.
    // TODO: support a shape with a shift relative to the body.
    pub(crate) shapes: GpuVector<GpuShape>,
    // TODO: it’s a bit weird that we store the vertex buffer but not the
    //       index buffer. This is because our only use-case currently
    //       is from wgsparkl which has its own way of storing indices.
    pub(crate) shapes_local_vertex_buffers: GpuVector<Point<f32>>,
    pub(crate) shapes_vertex_buffers: GpuVector<Point<f32>>,
    pub(crate) shapes_index_buffers: GpuVector<u32>,
    pub(crate) shapes_vertex_collider_id: GpuVector<u32>, // NOTE: this is a bit of a hack for wgsparkl
}

#[derive(Copy, Clone)]
/// Helper struct for defining a rigid-body to be added to a [`GpuBodySet`].
pub struct BodyDesc {
    /// The rigid-body’s mass-properties in local-space.
    pub local_mprops: GpuLocalMassProperties,
    /// The rigid-body’s mass-properties in world-space.
    pub mprops: GpuWorldMassProperties,
    /// The rigid-body’s linear and angular velocities.
    pub vel: GpuVelocity,
    /// The rigid-body’s world-space pose.
    pub pose: GpuSim,
    /// The rigid-body’s shape.
    pub shape: GpuShape,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            local_mprops: Default::default(),
            mprops: Default::default(),
            vel: Default::default(),
            pose: Default::default(),
            shape: GpuShape::cuboid(Vector::repeat(0.5)),
        }
    }
}

/// Coupling mode between a GPU body and the physics simulation.
///
/// This controls whether a body is affected by physics forces or acts as a kinematic body.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BodyCoupling {
    /// One-way coupling: the body affects other bodies but is not affected by them.
    ///
    /// This is useful for kinematic bodies that move independently of physics forces.
    OneWay,
    /// Two-way coupling: the body both affects and is affected by other bodies.
    ///
    /// This is the standard mode for dynamic rigid bodies.
    #[default]
    TwoWays,
}

/// Associates a Rapier body/collider pair with a coupling mode.
///
/// Used when creating a [`GpuBodySet`] from Rapier data structures to specify
/// which bodies should have two-way vs one-way coupling.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BodyCouplingEntry {
    /// The Rapier rigid body handle.
    pub body: RigidBodyHandle,
    /// The Rapier collider handle.
    pub collider: ColliderHandle,
    /// The coupling mode for this body.
    pub mode: BodyCoupling,
}

impl GpuBodySet {
    /// Returns `true` if this set contains no rigid bodies.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of rigid bodies in this set.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Creates a new GPU body set from Rapier rigid bodies and colliders.
    ///
    /// # Parameters
    ///
    /// - `device`: The WebGPU device for allocating GPU buffers.
    /// - `bodies`: Rapier rigid body set.
    /// - `colliders`: Rapier collider set.
    pub fn from_rapier(
        device: &Device,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        coupling: &[BodyCouplingEntry], // Only relevant to wgsparkl
    ) -> Self {
        let mut shape_buffers = ShapeBuffers::default();
        let mut gpu_bodies = vec![];
        let mut pt_collider_ids = vec![];

        for (co_id, coupling) in coupling.iter().enumerate() {
            let co = &colliders[coupling.collider];
            let rb = &bodies[coupling.body];

            let prev_len = shape_buffers.vertices.len();
            let shape = GpuShape::from_parry(co.shape(), &mut shape_buffers)
                .expect("Unsupported shape type");
            for _ in prev_len..shape_buffers.vertices.len() {
                pt_collider_ids.push(co_id as u32);
            }

            let zero_mprops = MassProperties::zero();
            let two_ways_coupling = rb.is_dynamic() && coupling.mode == BodyCoupling::TwoWays;
            let desc = BodyDesc {
                vel: GpuVelocity {
                    linear: *rb.linvel(),
                    #[allow(clippy::clone_on_copy)] // Needed for 2D/3D switch.
                    angular: rb.angvel().clone(),
                },
                #[cfg(feature = "dim2")]
                pose: (*rb.position()).into(),
                #[cfg(feature = "dim3")]
                pose: GpuSim3::from_isometry(*rb.position(), 1.0),
                shape,
                local_mprops: if two_ways_coupling {
                    rb.mass_properties().local_mprops.into()
                } else {
                    zero_mprops.into()
                },
                mprops: Default::default(),
            };
            gpu_bodies.push(desc);
        }

        Self::new(device, &gpu_bodies, &pt_collider_ids, &shape_buffers)
    }

    /// Create a set of `bodies` on the gpu.
    pub fn new(
        device: &Device,
        bodies: &[BodyDesc],
        pt_collider_ids: &[u32],
        shape_buffers: &ShapeBuffers,
    ) -> Self {
        #[allow(clippy::type_complexity)]
        let (local_mprops, (mprops, (vels, (poses, shapes_data)))): (
            Vec<_>,
            (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))),
        ) = bodies
            .iter()
            .copied()
            // NOTE: Looks silly, but we can’t just collect into (Vec, Vec, Vec).
            .map(|b| (b.local_mprops, (b.mprops, (b.vel, (b.pose, b.shape)))))
            .collect();
        // TODO: (api design) how can we let the user pick the buffer usages?
        Self {
            len: bodies.len() as u32,
            mprops: GpuVector::encase(device, &mprops, BufferUsages::STORAGE),
            local_mprops: GpuVector::encase(device, &local_mprops, BufferUsages::STORAGE),
            vels: GpuVector::encase(
                device,
                &vels,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            poses: GpuVector::init(
                device,
                &poses,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            shapes: GpuVector::init(device, &shapes_data, BufferUsages::STORAGE),
            shapes_local_vertex_buffers: GpuVector::encase(
                device,
                &shape_buffers.vertices,
                BufferUsages::STORAGE,
            ),
            shapes_vertex_buffers: GpuVector::encase(
                device,
                // TODO: init in world-space directly?
                &shape_buffers.vertices,
                BufferUsages::STORAGE,
            ),
            shapes_index_buffers: GpuVector::init(
                device,
                &shape_buffers.indices,
                BufferUsages::STORAGE,
            ),
            shapes_vertex_collider_id: GpuVector::init(
                device,
                pt_collider_ids,
                BufferUsages::STORAGE,
            ),
            shapes_data,
        }
    }

    /// GPU storage buffer containing the poses of every rigid-body.
    pub fn poses(&self) -> &GpuVector<GpuSim> {
        &self.poses
    }

    /// GPU storage buffer containing the velocities of every rigid-body.
    pub fn vels(&self) -> &GpuVector<GpuVelocity> {
        &self.vels
    }

    /// GPU storage buffer containing the world-space mass-properties of every rigid-body.
    pub fn mprops(&self) -> &GpuVector<GpuWorldMassProperties> {
        &self.mprops
    }

    /// GPU storage buffer containing the local-space mass-properties of every rigid-body.
    pub fn local_mprops(&self) -> &GpuVector<GpuLocalMassProperties> {
        &self.local_mprops
    }

    /// GPU storage buffer containing the shape of every rigid-body.
    pub fn shapes(&self) -> &GpuVector<GpuShape> {
        &self.shapes
    }

    /// Returns the GPU buffer containing shape vertices in world-space.
    ///
    /// This buffer is updated each frame as bodies move.
    pub fn shapes_vertex_buffers(&self) -> &GpuVector<Point<f32>> {
        &self.shapes_vertex_buffers
    }

    /// Returns the GPU buffer containing shape vertices in local-space.
    ///
    /// These are the original vertex positions before transformation.
    pub fn shapes_local_vertex_buffers(&self) -> &GpuVector<Point<f32>> {
        &self.shapes_local_vertex_buffers
    }

    /// Returns the GPU buffer mapping each vertex to its collider ID.
    ///
    /// This is used by wgsparkl for particle-body interactions.
    pub fn shapes_vertex_collider_id(&self) -> &GpuVector<u32> {
        &self.shapes_vertex_collider_id
    }

    /// Returns a CPU-side slice of the shape data.
    ///
    /// Useful for accessing shape information without GPU readback.
    pub fn shapes_data(&self) -> &[GpuShape] {
        &self.shapes_data
    }
}

#[derive(Shader)]
#[shader(
    derive(WgQuat, WgSim3, WgSim2),
    src = "body.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader defining structs related to rigid-bodies, as well as functions to compute point velocities
/// and update world-space mass-properties.
pub struct WgBody;

// TODO: this test won’t pass due to the lack of `substitute_aliases`
//       and `dim_shader_defs` in the macro. Figure out a way to make this work.
// wgcore::test_shader_compilation!(WgBody);
