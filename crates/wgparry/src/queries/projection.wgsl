//! Point Projection Result Module
//!
//! This module defines the common return type for all point projection operations.
//! `ProjectionResult` carries both the projected point location and information about
//! whether the original point was inside the shape.

#define_import_path wgparry::projection

/// The result of a point projection operation.
///
/// This structure is returned by all point projection functions.
struct ProjectionResult {
    /// The point’s projection on the shape.
    /// This can be equal to the original point if the point was inside
    /// of the shape and the projection function doesn’t always project
    /// on the boundary.
    point: Vector,
    /// Is the point inside of the shape?
    is_inside: bool,
}

struct ProjectionWithLocation {
    point: Vector,
    bcoords: Vector,
    // 0: vertex, 1: edge, 2: face
    feature_type: u32,
    id: u32,
    inside: bool,
}

fn vertex(pt: Vector, id: u32, inside: bool) -> ProjectionWithLocation {
    return ProjectionWithLocation(pt, Vector(), FEATURE_VERTEX, id, inside);
}

fn edge(pt: Vector, bcoords: vec2<f32>, id: u32, inside: bool) -> ProjectionWithLocation {
    return ProjectionWithLocation(pt, vec3(bcoords, 0.0), FEATURE_EDGE, id, inside);
}

fn face(pt: Vector, bcoords: vec3<f32>, id: u32, inside: bool) -> ProjectionWithLocation {
    return ProjectionWithLocation(pt, bcoords, FEATURE_FACE, id, inside);
}

fn solid(pt: Vector) -> ProjectionWithLocation {
    return ProjectionWithLocation(pt, Vector(), FEATURE_SOLID, 0, true);
}

#if DIM == 3
fn barycentric_coordinates(proj: ProjectionWithLocation) -> vec3<f32> {
    var bcoords = vec3(0.0);

    switch proj.feature_type {
        case FEATURE_VERTEX: {
            bcoords[proj.id] = 1.0;
        }
        case FEATURE_EDGE: {
            switch proj.id {
                case 0: {
                    bcoords[0] = proj.bcoords[0];
                    bcoords[1] = proj.bcoords[1];
                }
                case 1: {
                    bcoords[1] = proj.bcoords[0];
                    bcoords[2] = proj.bcoords[1];
                }
                case 2: {
                    bcoords[0] = proj.bcoords[0];
                    bcoords[2] = proj.bcoords[1];
                }
                default: { /* unreachable */ }
            }
        }
        case FEATURE_FACE: {
            bcoords = proj.bcoords;
        }
        default: { /* no valid barycentric coordinates */ }
    }

    return bcoords;
}
#endif

const FEATURE_VERTEX: u32 = 0;
const FEATURE_EDGE: u32 = 1;
const FEATURE_FACE: u32 = 2;
const FEATURE_SOLID: u32 = 3;

// TODO: move that to its own utility file
fn relative_eq(a: vec3<f32>, b: vec3<f32>) -> bool {
    const EPSILON: vec3<f32> = vec3(1.1920929E-7);

    let abs_diff = abs(a - b);

    // For when the numbers are really close together
    if all(abs_diff <= EPSILON) {
        return true;
    }

    let abs_a = abs(a);
    let abs_b = abs(b);

    // Use a relative difference comparison
    return all(abs_diff <= max(abs_b, abs_a) * EPSILON);
}