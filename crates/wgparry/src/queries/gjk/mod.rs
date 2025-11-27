use crate::queries::WgProjection;
use crate::shapes::{WgSegment, WgShape, WgTriangle};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;

#[cfg(feature = "dim3")]
use crate::shapes::WgTetrahedron;

#[derive(Shader)]
#[shader(
    src = "cso_point.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
pub struct WgCsoPoint;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgShape),
    src = "gjk.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
pub struct WgGjk;

#[derive(Shader)]
#[cfg_attr(
    feature = "dim2",
    shader(
        derive(WgCsoPoint, WgSegment, WgTriangle, WgProjection),
        src = "voronoi_simplex2.wgsl",
        shader_defs = "dim_shader_defs",
        src_fn = "substitute_aliases"
    )
)]
#[cfg_attr(
    feature = "dim3",
    shader(
        derive(WgCsoPoint, WgSegment, WgTriangle, WgTetrahedron, WgProjection),
        src = "voronoi_simplex3.wgsl",
        shader_defs = "dim_shader_defs",
        src_fn = "substitute_aliases"
    )
)]
pub struct WgVoronoiSimplex;

#[derive(Shader)]
#[cfg_attr(
    feature = "dim2",
    shader(
        derive(WgCsoPoint, WgVoronoiSimplex, WgShape, WgGjk),
        src = "epa2.wgsl",
        shader_defs = "dim_shader_defs",
        src_fn = "substitute_aliases"
    )
)]
#[cfg_attr(
    feature = "dim3",
    shader(
        derive(WgCsoPoint, WgVoronoiSimplex, WgShape, WgGjk),
        src = "epa3.wgsl",
        shader_defs = "dim_shader_defs",
        src_fn = "substitute_aliases"
    )
)]
pub struct WgEpa;

wgcore::test_shader_compilation!(WgVoronoiSimplex);
wgcore::test_shader_compilation!(WgGjk, wgcore, crate::dim_shader_defs());
wgcore::test_shader_compilation!(WgCsoPoint);
wgcore::test_shader_compilation!(WgEpa, wgcore, crate::dim_shader_defs());
