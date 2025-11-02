//! Bounding volume data structures for collision detection acceleration.
//!
//! Bounding volumes are simple geometric shapes that enclose more complex objects.
//! They enable fast broad-phase collision detection by providing cheap overlap tests
//! before performing expensive narrow-phase collision detection.
//!
//! Only Axis-Aligned Bounding Boxes (AABB) are provided at the moment.
mod aabb;

pub use aabb::*;
