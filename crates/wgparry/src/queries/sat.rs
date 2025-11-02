//! Separating Axis Theorem (SAT) collision detection.
//!
//! SAT is an efficient algorithm for detecting collisions between convex polyhedra
//! (like cuboids). It works by testing potential separating axes: if any axis exists
//! along which the shapes' projections don't overlap, the shapes don't collide.

use crate::shapes::WgCuboid;
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgCuboid, WgSim3, WgSim2),
    src = "sat.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// GPU shader implementing the Separating Axis Theorem for collision detection.
///
/// This shader provides SAT-based collision detection, particularly efficient for
/// detecting collisions between convex polyhedra like cuboids. SAT is a fundamental
/// algorithm in computational geometry that determines whether two convex shapes
/// overlap by testing separating planes.
pub struct WgSat;

wgcore::test_shader_compilation!(WgSat);
