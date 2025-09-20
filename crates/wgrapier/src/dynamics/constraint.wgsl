#define_import_path wgrapier::dynamics::constraint


#if DIM == 2
const SUB_LEN: u32 = 1;
#else
const SUB_LEN: u32 = 2;
#endif

// PERF: differentiate two-bodies and one-body constraints?
struct TwoBodyConstraint {
    dir_a: Vector, // Non-penetration force direction for the first body.
#if DIM == 3
    tangent_a: Vector, // One of the friction force directions.
#endif
    im_a: Vector,
    im_b: Vector,
    cfm_factor: f32,
    limit: f32,
    solver_vel_a: u32,
    solver_vel_b: u32,
    // TODO: support multiple constraint elements?
    elements: TwoBodyConstraintElement,
}

struct TwoBodyConstraintElement {
    normal_part: TwoBodyConstraintNormalPart,
    tangent_part: TwoBodyConstraintTangentPart,
}

struct TwoBodyConstraintNormalPart {
    gcross_a: AngVector,
    gcross_b: AngVector,
    rhs: f32,
    rhs_wo_bias: f32,
    impulse_a: f32,
    impulse_b: f32,
    r: f32,
    // For coupled constraint pairs, even constraints store the
    // diagonal of the projected mass matrix. Odd constraints
    // store the off-diagonal element of the projected mass matrix,
    // as well as the off-diagonal element of the inverse projected mass matrix.
    r_mat_elts: vec2<f32>,
}

struct TwoBodyConstraintTangentPart {
    gcross_a: array<AngVector, SUB_LEN>,
    gcross_b: array<AngVector, SUB_LEN>,
    rhs: array<f32, SUB_LEN>,
    rhs_wo_bias: array<f32, SUB_LEN>,

#if DIM == 2
    impulse: array<f32, 1>,
    impulse_accumulator: array<f32, 1>,
    r: array<f32, 1>,
#else
    impulse: array<f32, 2>,
    impulse_accumulator: array<f32, 2>,
    r: array<f32, 3>,
#endif
}