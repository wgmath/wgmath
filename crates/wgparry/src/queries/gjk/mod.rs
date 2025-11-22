use crate::queries::{
    specialize_cone_cone, specialize_cuboid_cone, specialize_cylinder_cone,
    specialize_cylinder_cuboid, specialize_cylinder_cylinder, WgProjection,
};
use crate::shapes::{WgCone, WgCuboid, WgCylinder, WgSegment, WgTetrahedron, WgTriangle};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;

#[derive(Shader)]
#[shader(
    src = "cso_point.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
pub struct WgCsoPoint;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgCuboid),
    src = "gjk.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
pub struct WgGjk;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgSegment, WgTriangle, WgTetrahedron, WgProjection),
    src = "voronoi_simplex3.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
pub struct WgVoronoiSimplex;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgCuboid),
    src = "epa3.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
pub struct WgEpaGeneric;

/*
 * Specializations.
 */
#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgCuboid, WgGjk),
    src = "epa3.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
pub struct WgEpa;

/*
 * Specialization through string substitution.
 * This is ugly, but kind of our only choice?
 */

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgCuboid, WgCylinder),
    src = "gjk.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cylinder_cuboid"
)]
pub struct WgGjkCylinderCuboid;

#[derive(Shader)]
#[shader(
    derive(
        WgCsoPoint,
        WgVoronoiSimplex,
        WgGjkCylinderCuboid,
        WgCuboid,
        WgCylinder
    ),
    src = "epa3.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cylinder_cuboid"
)]
pub struct WgEpaCylinderCuboid;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgCuboid, WgCylinder),
    src = "gjk.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cylinder_cylinder"
)]
pub struct WgGjkCylinderCylinder;
#[derive(Shader)]
#[shader(
    derive(
        WgCsoPoint,
        WgVoronoiSimplex,
        WgGjkCylinderCylinder,
        WgCuboid,
        WgCylinder
    ),
    src = "epa3.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cylinder_cylinder"
)]
pub struct WgEpaCylinderCylinder;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgCuboid, WgCylinder, WgCone),
    src = "gjk.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cylinder_cone"
)]
pub struct WgGjkCylinderCone;
#[derive(Shader)]
#[shader(
    derive(
        WgCsoPoint,
        WgVoronoiSimplex,
        WgGjkCylinderCone,
        WgCuboid,
        WgCylinder,
        WgCone
    ),
    src = "epa3.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cylinder_cone"
)]
pub struct WgEpaCylinderCone;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgCuboid, WgCylinder, WgCone),
    src = "gjk.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cuboid_cone"
)]
pub struct WgGjkCuboidCone;
#[derive(Shader)]
#[shader(
    derive(
        WgCsoPoint,
        WgVoronoiSimplex,
        WgGjkCuboidCone,
        WgCuboid,
        WgCylinder,
        WgCone
    ),
    src = "epa3.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cuboid_cone"
)]
pub struct WgEpaCuboidCone;

#[derive(Shader)]
#[shader(
    derive(WgCsoPoint, WgVoronoiSimplex, WgCuboid, WgCylinder, WgCone),
    src = "gjk.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cone_cone"
)]
pub struct WgGjkConeCone;
#[derive(Shader)]
#[shader(
    derive(
        WgCsoPoint,
        WgVoronoiSimplex,
        WgGjkCuboidCone,
        WgCuboid,
        WgCylinder,
        WgCone
    ),
    src = "epa3.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "specialize_cone_cone"
)]
pub struct WgEpaConeCone;

wgcore::test_shader_compilation!(WgVoronoiSimplex);
wgcore::test_shader_compilation!(WgGjk, wgcore, crate::dim_shader_defs());
wgcore::test_shader_compilation!(WgGjkCylinderCuboid, wgcore, crate::dim_shader_defs());
wgcore::test_shader_compilation!(WgCsoPoint);
wgcore::test_shader_compilation!(WgEpa, wgcore, crate::dim_shader_defs());
wgcore::test_shader_compilation!(WgEpaCylinderCuboid, wgcore, crate::dim_shader_defs());
