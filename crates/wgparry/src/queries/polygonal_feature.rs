//! Polygonal feature extraction and contact manifold generation.
//!
//! Polygonal features (faces, edges, vertices) are geometric primitives extracted from
//! shapes like cuboids for contact manifold generation. This module implements the
//! feature-clipping algorithm for generating multi-point contact manifolds in one shot.
//!
//! # Feature Clipping
//!
//! When two polyhedra collide, the algorithm:
//! 1. Identifies the colliding features (faces, edges, or vertices).
//! 2. Clips the incident feature against the reference feature's boundaries.
//! 3. Generates contact points where the features overlap.

use super::WgContactManifold;
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgContactManifold),
    src = "polygonal_feature.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// GPU shader for polygonal feature extraction and contact manifold generation.
///
/// This shader implements algorithms for:
/// - Extracting geometric features (faces, edges, vertices) from polygonal shapes.
/// - Clipping features against each other to find contact points.
/// - Generating contact manifolds with multiple points for stability.
///
/// # Feature Types
///
/// - **Face**: A flat polygonal surface (most common collision feature).
/// - **Edge**: A line segment (edge-edge contacts in 3D).
/// - **Vertex**: A point (vertex-face or vertex-edge contacts).
///
/// # Contact Generation Process
///
/// 1. Identify which features are colliding (face-face, edge-edge, etc.).
/// 2. Choose a reference feature and an incident feature.
/// 3. Clip the incident feature against the reference feature's boundaries.
/// 4. Keep points that penetrate the reference feature as contact points.
pub struct WgPolygonalFeature;

wgcore::test_shader_compilation!(WgPolygonalFeature, wgcore, crate::dim_shader_defs());
