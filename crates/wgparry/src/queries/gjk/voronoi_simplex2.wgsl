#define_import_path wgparry::gjk::voronoi_simplex

#import wgparry::gjk::cso_point as CsoPoint
#import wgparry::segment as Segment
#import wgparry::triangle as Triangle
#import wgparry::projection as Proj

/// A simplex of dimension up to 2 using Voronoï regions for computing point projections.
struct VoronoiSimplex {
    prev_vertices: array<u32, 3>,
    prev_proj: vec2<f32>,
    prev_dim: u32,

    vertices: array<CsoPoint::CsoPoint, 3>,
    proj: vec2<f32>,
    dim: u32,
}

struct SimplexProjectionResult {
    simplex: VoronoiSimplex,
    point: vec2<f32>,
}

fn init(pt: CsoPoint::CsoPoint) -> VoronoiSimplex {
    let origin = CsoPoint::origin();
    let prev_vertices = array(0u, 1u, 2u);
    let prev_proj = vec2(0.0, 0.0);
    let prev_dim = 0u;
    let vertices = array(pt, origin, origin);
    let proj = vec2(0.0, 0.0);
    let dim = 0u;
    return VoronoiSimplex(prev_vertices, prev_proj, prev_dim, vertices, proj, dim);
}

// TODO: all these free functions should be methods (not supported by wgsl).

fn reset(s: VoronoiSimplex, pt: CsoPoint::CsoPoint) -> VoronoiSimplex {
    return VoronoiSimplex(s.prev_vertices, s.prev_proj, 0u, array(pt, pt, pt), s.proj, 0u);
}


/// Add a point to this simplex.
fn add_point(s: VoronoiSimplex, pt: CsoPoint::CsoPoint) -> VoronoiSimplex {
    var result = s;
    result.prev_dim = s.dim;
    result.prev_proj = s.proj;
    result.prev_vertices = array(0, 1, 2);

    for (var i = 0u; i < s.dim + 1u; i++) {
        let dpt = s.vertices[i].point - pt.point;
        if dot(dpt, dpt) < CsoPoint::EPS_TOL {
            return s;
        }
    }

    result.dim += 1;
    result.vertices[result.dim] = pt;
    return result;
}

/// Projects the origin on the boundary of this simplex and reduces `s` the smallest subsimplex containing the origin.
///
/// Returns the result of the projection or Point::origin() if the origin lies inside of the simplex.
/// The state of the simplex before projection is saved, and can be retrieved using the methods prefixed
/// by `prev_`.
fn project_origin_and_reduce(s_: VoronoiSimplex) -> SimplexProjectionResult {
    var s = s_;

    switch s.dim {
        case 0: {
            s.proj[0] = 1.0;
            return SimplexProjectionResult(s, s.vertices[0].point);
        }
        case 1: {
            let seg = Segment::Segment(s.vertices[0u].point, s.vertices[1u].point);
            let proj = Segment::project_local_point_and_get_location(seg, vec2(), true);

            switch proj.feature_type {
                case Proj::FEATURE_VERTEX: {
                    if proj.id == 0u {
                        s.proj[0] = 1.0;
                        s.dim = 0;
                    } else {
                        // Swap 0, 1
                        let tmp = s.vertices[0];
                        s.vertices[0] = s.vertices[1];
                        s.vertices[1] = tmp;
                        let tmp_prev = s.prev_vertices[0];
                        s.prev_vertices[0] = s.prev_vertices[1];
                        s.prev_vertices[1] = tmp_prev;

                        s.proj[0] = 1.0;
                        s.dim = 0u;
                    }
                }
                case Proj::FEATURE_EDGE: {
                    s.proj[0] = proj.bcoords[0];
                    s.proj[1] = proj.bcoords[1];
                }
                default: { /* unreachable */ }
            }

            return SimplexProjectionResult(s, proj.point);
        }
        default: {
            // case 2
            let tri = Triangle::Triangle(
                s.vertices[0].point,
                s.vertices[1].point,
                s.vertices[2].point,
            );
            var proj = Triangle::project_local_point_and_get_location(tri, vec2(), true);

            switch proj.feature_type {
                case Proj::FEATURE_VERTEX: {
                    let i = proj.id;

                    // Swap 0, i
                    let tmp = s.vertices[0];
                    s.vertices[0] = s.vertices[i];
                    s.vertices[i] = tmp;
                    let tmp_prev = s.prev_vertices[0];
                    s.prev_vertices[0] = s.prev_vertices[i];
                    s.prev_vertices[i] = tmp_prev;

                    s.proj[0] = 1.0;
                    s.dim = 0u;
                }
                case Proj::FEATURE_EDGE: {
                    if proj.id == 0u {
                        s.proj[0] = proj.bcoords[0];
                        s.proj[1] = proj.bcoords[1];
                        s.dim = 1;
                    } else if proj.id == 1u {
                        // Swap 0, 2
                        let tmp = s.vertices[0];
                        s.vertices[0] = s.vertices[2];
                        s.vertices[2] = tmp;
                        let tmp_prev = s.prev_vertices[0];
                        s.prev_vertices[0] = s.prev_vertices[2];
                        s.prev_vertices[2] = tmp_prev;

                        s.proj[0] = proj.bcoords[1];
                        s.proj[1] = proj.bcoords[0];
                        s.dim = 1u;
                    } else {
                        // Swap 1, 2
                        let tmp = s.vertices[1];
                        s.vertices[1] = s.vertices[2];
                        s.vertices[2] = tmp;
                        let tmp_prev = s.prev_vertices[1];
                        s.prev_vertices[1] = s.prev_vertices[2];
                        s.prev_vertices[2] = tmp_prev;

                        s.proj[0] = proj.bcoords[0];
                        s.proj[1] = proj.bcoords[1];
                        s.dim = 1u;
                    }
                }
                default: { /* unreachable */ }
            }

            return SimplexProjectionResult(s, proj.point);
        }
    }
}
