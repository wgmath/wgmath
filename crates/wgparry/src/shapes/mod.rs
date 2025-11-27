//! Geometric shape definitions and their GPU shader implementations.
//!
//! This module provides GPU-accelerated geometric primitives used for collision detection
//! and physics simulation.

mod ball;
mod capsule;
mod convex_polyhedron;
mod cuboid;
mod polyline;
mod segment;
mod triangle;

#[cfg(feature = "dim3")]
mod cone;
#[cfg(feature = "dim3")]
mod cylinder;
#[cfg(feature = "dim3")]
mod tetrahedron;

mod shape;
mod trimesh;
mod vtx_idx;

pub use ball::*;
pub use capsule::*;
pub use convex_polyhedron::*;
pub use cuboid::*;
pub use polyline::*;
pub use segment::*;
pub use shape::*;
pub use triangle::*;
pub use trimesh::*;
pub use vtx_idx::*;

#[cfg(feature = "dim3")]
pub use cone::*;
#[cfg(feature = "dim3")]
pub use cylinder::*;
#[cfg(feature = "dim3")]
pub use tetrahedron::*;
