//! Geometric shape definitions and their GPU shader implementations.
//!
//! This module provides GPU-accelerated geometric primitives used for collision detection
//! and physics simulation.

mod ball;
mod capsule;
mod cuboid;
mod segment;
mod tetrahedron;
mod triangle;

#[cfg(feature = "dim3")]
mod cone;
#[cfg(feature = "dim3")]
mod cylinder;
mod shape;

pub use ball::*;
pub use capsule::*;
pub use cuboid::*;
pub use segment::*;
pub use shape::*;
pub use tetrahedron::*;
pub use triangle::*;

#[cfg(feature = "dim3")]
pub use cone::*;
#[cfg(feature = "dim3")]
pub use cylinder::*;
