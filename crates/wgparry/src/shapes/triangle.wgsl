//! Triangle Shape Module
//!
//! This module provides the triangle shape definition from its three vertices.

#define_import_path wgparry::triangle

#import wgparry::projection as Proj

#if DIM == 2
const DIM: u32 = 2;
#else
const DIM: u32 = 3;
#endif

/// A triangle defined by three vertices.
struct Triangle {
    /// First vertex of the triangle.
    a: Vector,
    /// Second vertex of the triangle.
    b: Vector,
    /// Third vertex of the triangle.
    c: Vector,
}

fn is_proj_inside(pt: Vector, proj: Vector) -> bool {
#if DIM == 2
        return proj == pt;
#else
        // TODO: is this acceptable to assume the point is inside of the
        // triangle if it is close enough?
        return Proj::relative_eq(proj, pt);
#endif
}

const AB: u32 = 0;
const AC: u32 = 1;
const BC: u32 = 2;
const FACE_CW: u32 = 3;
const FACE_CCW: u32 = 4;
const FACE_INTERIOR: u32 = 5; // 2D only

struct ProjectionInfo {
    feature: u32,
    params: vec3<f32>,
}

// Checks on which edge voronoï region the point is.
// For 2D and 3D, it uses explicit cross/perp products that are
// more numerically stable.
fn stable_check_edges_voronoi(
    ab: Vector,
    ac: Vector,
    bc: Vector,
    ap: Vector,
    bp: Vector,
    cp: Vector,
    ab_ap: f32,
    ab_bp: f32,
    ac_ap: f32,
    ac_cp: f32,
    ac_bp: f32,
    ab_cp: f32,
) -> ProjectionInfo {
#if DIM == 2
    let n = perp(ab, ac);
    let vc = n * perp(ab, ap);
    if vc < 0.0 && ab_ap >= 0.0 && ab_bp <= 0.0 {
        return ProjectionInfo(AB, Vector());
    }

    let vb = -n * perp(ac, cp);
    if vb < 0.0 && ac_ap >= 0.0 && ac_cp <= 0.0 {
        return ProjectionInfo(AC, Vector());
    }

    let va = n * perp(bc, bp);
    if va < 0.0 && ac_bp - ab_bp >= 0.0 && ab_cp - ac_cp >= 0.0 {
        return ProjectionInfo(BC, Vector());
    }

    ProjectionInfo(0, va, vb, vc)
#else
    let n = cross(ab, ac);
    let vc = dot(n, cross(ab, ap));
    if vc < 0.0 && ab_ap >= 0.0 && ab_bp <= 0.0 {
        return ProjectionInfo(AB, Vector());
    }

    let vb = -dot(n, cross(ac, cp));
    if vb < 0.0 && ac_ap >= 0.0 && ac_cp <= 0.0 {
        return ProjectionInfo(AC, Vector());
    }

    let va = dot(n, cross(bc, bp));
    if va < 0.0 && ac_bp - ab_bp >= 0.0 && ab_cp - ac_cp >= 0.0 {
        return ProjectionInfo(BC, Vector());
    }

    if dot(n, ap) >= 0.0 {
        return ProjectionInfo(FACE_CW, vec3(va, vb, vc));
    } else {
        return ProjectionInfo(FACE_CCW, vec3(va, vb, vc));
    }
#endif
}

fn project_local_point_and_get_location(shape: Triangle, pt: Vector, solid: bool) -> Proj::ProjectionWithLocation {
    // To understand the ideas, consider reading the slides below
    // https://box2d.org/files/ErinCatto_GJK_GDC2010.pdf
    let a = shape.a;
    let b = shape.b;
    let c = shape.c;

    let ab = b - a;
    let ac = c - a;
    let ap = pt - a;

    let ab_ap = dot(ab, ap);
    let ac_ap = dot(ac, ap);

    if ab_ap <= 0.0 && ac_ap <= 0.0 {
        // Voronoï region of `a`.
        let inside = is_proj_inside(pt, a);
        return Proj::vertex(a, 0, inside);
    }

    let bp = pt - b;
    let ab_bp = dot(ab, bp);
    let ac_bp = dot(ac, bp);

    if ab_bp >= 0.0 && ac_bp <= ab_bp {
        // Voronoï region of `b`.
        let inside = is_proj_inside(pt, b);
        return Proj::vertex(b, 1, inside);
    }

    let cp = pt - c;
    let ab_cp = dot(ab, cp);
    let ac_cp = dot(ac, cp);

    if ac_cp >= 0.0 && ab_cp <= ac_cp {
        // Voronoï region of `c`.
        let inside = is_proj_inside(pt, c);
        return Proj::vertex(b, 2, inside);
    }

    let bc = c - b;
    let proj = stable_check_edges_voronoi(
        ab, ac, bc, ap, bp, cp, ab_ap, ab_bp, ac_ap, ac_cp, ac_bp, ab_cp,
    );

    switch proj.feature {
       case AB: {
           // Voronoï region of `ab`.
           let v = ab_ap / dot(ab, ab);
           let bcoords = vec2(1.0 - v, v);
           let res = a + ab * v;
           return Proj::edge(res, bcoords, 0, is_proj_inside(pt, res));
       }
       case AC: {
           // Voronoï region of `ac`.
           let w = ac_ap / dot(ac, ac);
           let bcoords = vec2(1.0 - w, w);
           let res = a + ac * w;
           return Proj::edge(res, bcoords, 2, is_proj_inside(pt, res));
       }
       case BC: {
           // Voronoï region of `bc`.
           let w = dot(bc, bp) / dot(bc, bc);
           let bcoords = vec2(1.0 - w, w);
           let res = b + bc * w;
           return Proj::edge(res, bcoords, 1, is_proj_inside(pt, res));
       }
       case FACE_CW, FACE_CCW: {
           // Voronoï region of the face.
           if DIM != 2 {
               // NOTE: in some cases, numerical instability
               // may result in the denominator being zero
               // when the triangle is nearly degenerate.
               if proj.params.x + proj.params.y + proj.params.z != 0.0 {
                   let denom = 1.0 / (proj.params.x + proj.params.y + proj.params.z);
                   let v = proj.params.y * denom;
                   let w = proj.params.z * denom;
                   let bcoords = vec3(1.0 - v - w, v, w);
                   let res = a + ab * v + ac * w;
                   return Proj::face(res, bcoords, proj.feature, is_proj_inside(pt, res));
               }
           }
       }
       default: { /* FACE_INTERIOR, 2D only (implemented below) */ }
    }

    // Special treatment if we work in 2d because in this case we really are inside of the
    // object.
    if solid {
        return Proj::solid(pt);
    } else {
        // We have to project on the closest edge.

        // TODO: this might be optimizable.
        // TODO: be careful with numerical errors.
        let v = ab_ap / (ab_ap - ab_bp); // proj on ab = a + ab * v
        let w = ac_ap / (ac_ap - ac_cp); // proj on ac = a + ac * w
        let u = (ac_bp - ab_bp) / (ac_bp - ab_bp + ab_cp - ac_cp); // proj on bc = b + bc * u

        let bc = c - b;
        let d_ab = dot(ap, ap) - (dot(ab, ab) * v * v);
        let d_ac = dot(ap, ap) - (dot(ac, ac) * w * w);
        let d_bc = dot(bp, bp) - (dot(bc, bc) * u * u);

        if d_ab < d_ac {
            if d_ab < d_bc {
                // ab
                let bcoords = vec2(1.0 - v, v);
                let proj = a + ab * v;
                return Proj::edge(proj, bcoords, 0, true);
            } else {
                // bc
                let bcoords = vec2(1.0 - u, u);
                let proj = b + bc * u;
                return Proj::edge(proj, bcoords, 1, true);
            }
        } else if d_ac < d_bc {
            // ac
            let bcoords = vec2(1.0 - w, w);
            let proj = a + ac * w;
            return Proj::edge(proj, bcoords, 2, true);
        } else {
            // bc
            let bcoords = vec2(1.0 - u, u);
            let proj = b + bc * u;
            return Proj::edge(proj, bcoords, 1, true);
        }
    }
}