//! Tetrahedron shape.
//!
//! A tetrahedron is defined by three vertices (A, B, C).

use crate::queries::WgProjection;
use crate::substitute_aliases;
use wgcore::Shader;

#[derive(Shader)]
#[shader(
    derive(WgProjection),
    src = "tetrahedron.wgsl",
    src_fn = "substitute_aliases"
)]
/// GPU shader for the tetrahedron shape.
pub struct WgTetrahedron;
