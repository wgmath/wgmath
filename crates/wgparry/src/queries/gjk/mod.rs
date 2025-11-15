use crate::queries::WgProjection;
use crate::shapes::{WgCuboid, WgSegment, WgTetrahedron, WgTriangle};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;

#[derive(Shader)]
#[shader(
    src = "cso_point.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
struct WgCsoPoint;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgCuboid),
    src = "gjk.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
struct WgGjk;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgSegment, WgTriangle, WgTetrahedron, WgProjection),
    src = "voronoi_simplex3.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
struct WgVoronoiSimplex;

wgcore::test_shader_compilation!(WgVoronoiSimplex);
wgcore::test_shader_compilation!(WgGjk, wgcore, crate::dim_shader_defs());
wgcore::test_shader_compilation!(WgCsoPoint);
