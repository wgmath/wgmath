#define_import_path wgparry::gjk::voronoi_simplex

#import wgparry::gjk::cso_point as CsoPoint
#import wgparry::segment as Segment
#import wgparry::triangle as Triangle
#import wgparry::tetrahedron as Tetrahedron
#import wgparry::projection as Proj

/// A simplex of dimension up to 3 that uses Voronoï regions for computing point projections.
struct VoronoiSimplex {
    prev_vertices: array<u32, 4>,
    prev_proj: vec3<f32>,
    prev_dim: u32,

    vertices: array<CsoPoint::CsoPoint, 4>,
    proj: vec3<f32>,
    dim: u32,
}

struct SimplexProjectionResult {
    simplex: VoronoiSimplex,
    point: vec3<f32>,
}

fn init(pt: CsoPoint::CsoPoint) -> VoronoiSimplex {
    let origin = CsoPoint::origin();
    let prev_vertices = array(0u, 1u, 2u, 3u);
    let prev_proj = vec3(0.0, 0.0, 0.0);
    let prev_dim = 0u;
    let vertices = array(pt, origin, origin, origin);
    let proj = vec3(0.0, 0.0, 0.0);
    let dim = 0u;
    return VoronoiSimplex(prev_vertices, prev_proj, prev_dim, vertices, proj, dim);
}

fn reset(s: VoronoiSimplex, pt: CsoPoint::CsoPoint) -> VoronoiSimplex {
    return VoronoiSimplex(s.prev_vertices, s.prev_proj, 0u, array(pt, pt, pt, pt), s.proj, 0u);
}

fn add_point(s: VoronoiSimplex, pt: CsoPoint::CsoPoint) -> VoronoiSimplex {
    if s.dim == 0 {
        let dpt = s.vertices[0].point - pt.point;
        if dot(dpt, dpt) < CsoPoint::EPS_TOL {
            return s;
        }
    } else if s.dim == 1 {
        let ab = s.vertices[1].point - s.vertices[0].point;
        let ac = pt.point - s.vertices[0].point;
        let ab_ac = cross(ab, ac);
        if dot(ab_ac, ab_ac) < CsoPoint::EPS_TOL {
            return s;
        }
    } else { // s.dim == 2
        let ab = s.vertices[1].point - s.vertices[0].point;
        let ac = s.vertices[2].point - s.vertices[0].point;
        let ap = pt.point - s.vertices[0].point;
        let ab_ac = cross(ab, ac);
        let n = ab_ac / length(ab_ac);

        if abs(dot(n, ap)) < CsoPoint::EPS_TOL {
            return s;
        }
    }

    var result = s;
    result.prev_dim = s.dim;
    result.prev_proj = s.proj;
    result.prev_vertices = array(0, 1, 2, 3);
    result.dim += 1;
    result.vertices[result.dim] = pt;
    return result;
}

/// Projects the origin on the boundary of this simplex and reduces `self` the smallest subsimplex containing the origin.
///
/// Returns the result of the projection or `Point::origin()` if the origin lies inside of the simplex.
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
            let proj = Segment::project_local_point_and_get_location(seg, vec3(), true);

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
        case 2: {
            let tri = Triangle::Triangle(
                s.vertices[0].point,
                s.vertices[1].point,
                s.vertices[2].point,
            );
            var proj = Triangle::project_local_point_and_get_location(tri, vec3(), true);

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
                case Proj::FEATURE_FACE: {
                    s.proj = proj.bcoords;
                }
                default: { /* unreachable */ }
            }

            return SimplexProjectionResult(s, proj.point);
        }
        default: {
            let tetra = Tetrahedron::Tetrahedron(
                s.vertices[0].point,
                s.vertices[1].point,
                s.vertices[2].point,
                s.vertices[3].point,
            );
            let proj = Tetrahedron::project_local_point_and_get_location(tetra, vec3(), true);

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
                    s.dim = 0;
                }
                case Proj::FEATURE_EDGE: {
                    switch proj.id {
                        case 0: {
                            // ab
                        }
                        case 1: {
                            // ac
                            // Swap 1, 2
                            let tmp = s.vertices[1];
                            s.vertices[1] = s.vertices[2];
                            s.vertices[2] = tmp;
                            let tmp_prev = s.prev_vertices[1];
                            s.prev_vertices[1] = s.prev_vertices[2];
                            s.prev_vertices[2] = tmp_prev;
                        }
                        case 2: {
                            // ad
                            // Swap 1, 3
                            let tmp = s.vertices[1];
                            s.vertices[1] = s.vertices[3];
                            s.vertices[3] = tmp;
                            let tmp_prev = s.prev_vertices[1];
                            s.prev_vertices[1] = s.prev_vertices[3];
                            s.prev_vertices[3] = tmp_prev;
                        }
                        case 3: {
                            // bc
                            // Swap 0, 2
                            let tmp = s.vertices[0];
                            s.vertices[0] = s.vertices[2];
                            s.vertices[2] = tmp;
                            let tmp_prev = s.prev_vertices[0];
                            s.prev_vertices[0] = s.prev_vertices[2];
                            s.prev_vertices[2] = tmp_prev;
                        }
                        case 4: {
                            // bd
                            // Swap 0, 3
                            let tmp = s.vertices[0];
                            s.vertices[0] = s.vertices[3];
                            s.vertices[3] = tmp;
                            let tmp_prev = s.prev_vertices[0];
                            s.prev_vertices[0] = s.prev_vertices[3];
                            s.prev_vertices[3] = tmp_prev;
                        }
                        case 5: {
                            // cd
                            // Swap 0, 2
                            let tmp = s.vertices[0];
                            s.vertices[0] = s.vertices[2];
                            s.vertices[2] = tmp;
                            let tmp_prev = s.prev_vertices[0];
                            s.prev_vertices[0] = s.prev_vertices[2];
                            s.prev_vertices[2] = tmp_prev;

                            // Swap 1, 3
                            let tmp_ = s.vertices[1];
                            s.vertices[1] = s.vertices[3];
                            s.vertices[3] = tmp_;
                            let tmp_prev_ = s.prev_vertices[1];
                            s.prev_vertices[1] = s.prev_vertices[3];
                            s.prev_vertices[3] = tmp_prev_;
                        }
                        default: { /* unreachable */ }
                    }

                    switch proj.id {
                        case 0, 1, 2, 5: {
                            s.proj[0] = proj.bcoords[0];
                            s.proj[1] = proj.bcoords[1];
                        }
                        case 3, 4: {
                            s.proj[0] = proj.bcoords[1];
                            s.proj[1] = proj.bcoords[0];
                        }
                        default: { /* unreachable */ }
                    }
                    s.dim = 1;
                }
                case Proj::FEATURE_FACE: {
                    switch proj.id {
                        case 0: {
                            // abc
                            s.proj = proj.bcoords;
                        }
                        case 1: {
                            // abd
                            s.vertices[2] = s.vertices[3];
                            s.proj = proj.bcoords;
                        }
                        case 2: {
                            // acd
                            s.vertices[1] = s.vertices[3];
                            s.proj[0] = proj.bcoords[0];
                            s.proj[1] = proj.bcoords[2];
                            s.proj[2] = proj.bcoords[1];
                        }
                        case 3: {
                            // bcd
                            s.vertices[0] = s.vertices[3];
                            s.proj[0] = proj.bcoords[2];
                            s.proj[1] = proj.bcoords[0];
                            s.proj[2] = proj.bcoords[1];
                        }
                        default: { /* unreachable */ }
                    }

                    s.dim = 2;
                }
                default: { /* unreachable */ }
            }

            return SimplexProjectionResult(s, proj.point);
        }
    }

    // NOTE: we should never reach this.
    return SimplexProjectionResult();
}
