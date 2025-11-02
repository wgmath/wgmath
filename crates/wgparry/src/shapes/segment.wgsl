//! Segment (Line Segment) Module
//!
//! This module provides geometric operations for line segments.
//! A segment is defined by two endpoints and represents the straight line
//! connecting them.

#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::ray as Ray
#import wgparry::projection as Proj

#define_import_path wgparry::segment

/// A line segment defined by two endpoints.
struct Segment {
    /// First endpoint of the segment.
    a: Vector,
    /// Second endpoint of the segment.
    b: Vector,
}

/// Projects a point onto a line segment in local coordinates.
///
/// The projection uses Voronoi regions to handle the three cases:
/// 1. Point projects to endpoint 'a' (before the segment).
/// 2. Point projects to endpoint 'b' (after the segment).
/// 3. Point projects to the segment interior.
///
/// Parameters:
/// - seg: The line segment.
/// - pt: The point to project.
///
/// Returns: The closest point on the segment to pt.
/// TODO: implement the other projection functions
fn projectLocalPoint(seg: Segment, pt: Vector) -> Vector {
    let ab = seg.b - seg.a;
    let ap = pt - seg.a;
    let ab_ap = dot(ab, ap);
    let sqnab = dot(ab, ab);

    // PERF: would it be faster to do a bunch of `select` instead of `if`?
    if ab_ap <= 0.0 {
        // Voronoï region of vertex 'a'.
        return seg.a;
    } else if ab_ap >= sqnab {
        // Voronoï region of vertex 'b'.
        return seg.b;
    } else {
        // Voronoï region of the segment interior.
        let u = ab_ap / sqnab;
        return seg.a + ab * u;
    }
}
