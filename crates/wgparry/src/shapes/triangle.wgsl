//! Triangle Shape Module
//!
//! This module provides the triangle shape definition from its three vertices.

#define_import_path wgparry::triangle

/// A triangle defined by three vertices.
struct Triangle {
    /// First vertex of the triangle.
    a: Vector,
    /// Second vertex of the triangle.
    b: Vector,
    /// Third vertex of the triangle.
    c: Vector,
}
