//! Geometric query operations for collision detection and physics simulation.
//!
//! This module provides GPU-accelerated geometric algorithms for:
//! - **Ray-casting**: Finding intersections between rays and shapes
//! - **Point projection**: Finding the closest point on a shape's surface
//! - **Contact generation**: Computing contact points and normals for collision response
//! - **Separating axis tests**: Efficient collision detection between convex shapes

mod contact;
mod gjk;
mod polygonal_feature;
mod projection;
mod ray;
mod sat;
mod specializations;

pub use contact::*;
pub use polygonal_feature::*;
pub use projection::*;
pub use ray::*;
pub use sat::*;
pub use specializations::*;
