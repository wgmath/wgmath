//! Axis-Aligned Bounding Box (AABB) implementation.
//!
//! An AABB is the simplest and most widely used bounding volume, defined by its
//! minimum and maximum corner points along each coordinate axis. AABBs are not
//! rotated with the objects they bound; instead, they expand to accommodate rotation.
//!
//! # Properties
//!
//! - **Axis-aligned**: Edges are always parallel to coordinate axes.
//! - **Conservative**: Always contains the entire object (may have empty space).
//! - **Fast overlap test**: Just compare min/max coordinates (6 comparisons in 3D).
//! - **Fast to compute**: For most shapes, AABB computation is very efficient.

use crate::shapes::WgShape;
use crate::substitute_aliases;
use wgcore::{test_shader_compilation, Shader};
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgShape),
    src = "./aabb.wgsl",
    src_fn = "substitute_aliases"
)]
/// GPU shader for computing and manipulating Axis-Aligned Bounding Boxes.
///
/// This shader provides functions for:
/// - Computing AABBs from shapes.
/// - AABB overlap testing.
/// - AABB merging and manipulation.
pub struct WgAabb;

test_shader_compilation!(WgAabb);
