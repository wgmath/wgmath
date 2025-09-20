use encase::ShaderType;
use nalgebra::Vector2;
use wgcore::Shader;
use wgparry::math::{AngVector, Vector};
use wgparry::{dim_shader_defs, queries::WgContact, substitute_aliases};

#[derive(Shader)]
#[shader(
    derive(WgContact),
    src = "constraint.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
pub struct WgConstraint;

#[cfg(feature = "dim2")]
const SUB_LEN: usize = 1;
#[cfg(feature = "dim3")]
const SUB_LEN: usize = 2;

// PERF: differentiate two-bodies and one-body constraints?
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraint {
    pub dir_a: Vector<f32>, // Non-penetration force direction for the first body.
    #[cfg(feature = "dim3")]
    pub tangent_a: Vector<f32>, // One of the friction force directions.
    pub im_a: Vector<f32>,
    pub im_b: Vector<f32>,
    pub cfm_factor: f32,
    pub limit: f32,
    pub solver_vel_a: u32,
    pub solver_vel_b: u32,
    // TODO: support multiple constraint elements?
    pub elements: GpuTwoBodyConstraintElement,
}

#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraintElement {
    pub normal_part: GpuTwoBodyConstraintNormalPart,
    pub tangent_part: GpuTwoBodyConstraintTangentPart,
}

#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraintNormalPart {
    pub gcross_a: AngVector<f32>,
    pub gcross_b: AngVector<f32>,
    pub rhs: f32,
    pub rhs_wo_bias: f32,
    pub impulse: f32,
    pub impulse_accumulator: f32,
    pub r: f32,
    // For coupled constraint pairs, even constraints store the
    // diagonal of the projected mass matrix. Odd constraints
    // store the off-diagonal element of the projected mass matrix,
    // as well as the off-diagonal element of the inverse projected mass matrix.
    pub r_mat_elts: Vector2<f32>,
}

#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraintTangentPart {
    pub gcross_a: [AngVector<f32>; SUB_LEN],
    pub gcross_b: [AngVector<f32>; SUB_LEN],
    pub rhs: [AngVector<f32>; SUB_LEN],
    pub rhs_wo_bias: [AngVector<f32>; SUB_LEN],

    #[cfg(feature = "dim2")]
    pub impulse: [f32; 1],
    #[cfg(feature = "dim2")]
    pub impulse_accumulator: [f32; 1],
    #[cfg(feature = "dim2")]
    pub r: [f32; 1],
    #[cfg(feature = "dim3")]
    pub impulse: [f32; 2],
    #[cfg(feature = "dim3")]
    pub impulse_accumulator: [f32; 2],
    #[cfg(feature = "dim3")]
    pub r: [f32; 3],
}
