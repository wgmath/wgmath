#[cfg(feature = "dim2")]
use rapier2d as rapier;
#[cfg(feature = "dim3")]
use rapier3d as rapier;
#[cfg(feature = "dim2")]
use wgrapier2d as wgrapier;
#[cfg(feature = "dim3")]
use wgrapier3d as wgrapier;

use super::SimulationBackend;
use crate::SimulationState;
#[cfg(feature = "dim2")]
use nalgebra::Similarity2;
#[cfg(feature = "dim3")]
use nalgebra::Similarity3;
#[cfg(feature = "dim3")]
use rapier::dynamics::FrictionModel;
use rapier::dynamics::{CCDSolver, IntegrationParameters, IslandManager};
use rapier::geometry::{BroadPhaseBvh, ColliderSet, NarrowPhase};
use rapier::prelude::{ImpulseJointSet, MultibodyJointSet, PhysicsPipeline, RigidBodySet};
use wgcore::gpu::GpuInstance;
use wgcore::timestamps::GpuTimestamps;
use wgrapier::pipeline::RunStats;
use wgrapier::wgparry::math::GpuSim;

/// CPU-based physics backend using rapier
pub struct CpuBackend {
    pipeline: PhysicsPipeline,
    integration_parameters: IntegrationParameters,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    poses_cache: Vec<GpuSim>,
}

impl CpuBackend {
    pub fn new(phys: SimulationState) -> Self {
        let mut poses_cache = Vec::new();
        let mut shapes_cache = Vec::new();

        // Build initial poses and shapes from the simulation state
        for (_, co) in phys.colliders.iter() {
            #[cfg(feature = "dim2")]
            {
                poses_cache.push(GpuSim {
                    similarity: Similarity2::from_isometry(*co.position(), 1.0),
                    padding: Default::default(),
                });
            }
            #[cfg(feature = "dim3")]
            {
                poses_cache.push(Similarity3::from_isometry(*co.position(), 1.0));
            }
            shapes_cache.push(co.shared_shape().clone());
        }

        #[allow(unused_mut)] // mut not needed in 2D but needed in 3d.
        let mut params = IntegrationParameters::default();
        // NOTE: to keep the comparison fair, use the same friction model as the GPU version
        //       (the GPU doesn’t implement  twist friction yet).
        #[cfg(feature = "dim3")]
        {
            params.friction_model = FrictionModel::Coulomb;
        }

        Self {
            pipeline: PhysicsPipeline::new(),
            integration_parameters: params,
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: phys.bodies,
            colliders: phys.colliders,
            impulse_joints: phys.impulse_joints,
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            poses_cache,
        }
    }
}

impl SimulationBackend for CpuBackend {
    fn poses(&self) -> &[GpuSim] {
        &self.poses_cache
    }

    async fn step(
        &mut self,
        _gpu: Option<&GpuInstance>,
        _timestamps: Option<&mut GpuTimestamps>,
    ) -> RunStats {
        let t0 = web_time::Instant::now();

        self.pipeline.step(
            &(rapier::math::Vector::y() * -9.81),
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );
        let total_sim_time = t0.elapsed();

        // Update poses cache
        self.poses_cache.clear();
        for (_, co) in self.colliders.iter() {
            #[cfg(feature = "dim2")]
            {
                self.poses_cache.push(GpuSim {
                    similarity: Similarity2::from_isometry(*co.position(), 1.0),
                    padding: Default::default(),
                });
            }
            #[cfg(feature = "dim3")]
            {
                self.poses_cache
                    .push(Similarity3::from_isometry(*co.position(), 1.0));
            }
        }

        RunStats {
            total_simulation_time_with_readback: total_sim_time,
            ..Default::default()
        }
    }
}
