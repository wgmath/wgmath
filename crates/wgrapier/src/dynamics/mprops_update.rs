//! Rigid-bodies world-space mass properties calculation.

use crate::dynamics::{WgBody, WgSimParams};
use wgcore::Shader;
use wgparry::{dim_shader_defs, substitute_aliases};
use wgpu::ComputePipeline;

/// GPU shader for updating the world-space mass properties of rigid-bodies.
#[derive(Shader)]
#[shader(
    derive(WgBody, WgSimParams),
    src = "mprops_update.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs",
    composable = false
)]
pub struct WgMpropsUpdate {
    /// Compute pipeline for the gravity application kernel.
    ///
    /// Expected bind group layout:
    /// - Group 0: World mass properties, local mass properties, poses (storage buffers)
    pub main: ComputePipeline,
}
