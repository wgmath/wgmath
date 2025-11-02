//! Broad-phase collision detection algorithms for identifying potentially colliding pairs.
//!
//! Broad-phase collision detection quickly filters out non-colliding object pairs before
//! running expensive narrow-phase tests. This module provides GPU-accelerated implementations
//! of various broad-phase algorithms.
//!
//! # Available Algorithms
//!
//! ## Brute Force
//!
//! Tests all pairs of objects (O(n²) complexity). Simple but only practical for small scenes
//! (typically < 100 objects). It is mostly relevant for testing and debugging.
//!
//! **Pros**: Simple, no preprocessing, deterministic.
//! **Cons**: O(n²) scaling, impractical for large scenes.
//!
//! ## LBVH - Linear Bounding Volume Hierarchy
//!
//! Builds a binary tree of bounding volumes using Morton codes for spatial sorting.
//! Near-linear construction time (O(n log n)) and efficient traversal make it suitable
//! for large dynamic scenes.
//!
//! **Pros**: O(n log n) construction and O(log n) traversal (average), good for dynamic scenes.
//! **Cons**: Requires tree rebuild for moving objects, more complex than brute force.

mod brute_force_broad_phase;
mod lbvh;
mod narrow_phase;

pub use brute_force_broad_phase::*;
pub use lbvh::*;
pub use narrow_phase::*;
