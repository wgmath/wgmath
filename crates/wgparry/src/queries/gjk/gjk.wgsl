#define_import_path wgparry::gjk::gjk

#if DIM == 2
    #import wgebra::sim2 as Pose
    const DIM: u32 = 2;
#else
    #import wgebra::sim3 as Pose
    const DIM: u32 = 3;
#endif

#import wgparry::gjk::voronoi_simplex as VoronoiSimplex
#import wgparry::gjk::cso_point as CsoPoint
#import wgparry::cuboid as ShapeA
#import wgparry::cuboid as ShapeB

const INTERSECTION: u32 = 0;
const CLOSEST_POINTS: u32 = 1;
const PROXIMITY: u32 = 2;
const NO_INTERSECTION: u32 = 3;

struct GjkResult {
    status: u32,
    a: Vector,
    b: Vector,
    dir: Vector,
}

fn gjk_result_intersection() -> GjkResult {
    return GjkResult(INTERSECTION, Vector(), Vector(), Vector());
}

fn gjk_result_closest_points(a: Vector, b: Vector, dir: Vector) -> GjkResult {
    return GjkResult(CLOSEST_POINTS, a, b, dir);
}

fn gjk_result_proximity(dir: Vector) -> GjkResult {
    return GjkResult(PROXIMITY, Vector(), Vector(), dir);
}

fn gjk_result_no_intersection(dir: Vector) -> GjkResult {
    return GjkResult(NO_INTERSECTION, Vector(), Vector(), dir);
}

fn closest_points(
    pose12: Transform,
    g1: ShapeA::Cuboid,
    g2: ShapeB::Cuboid,
    max_dist: f32,
    exact_dist: bool,
    simplex: ptr<function, VoronoiSimplex::VoronoiSimplex>,
) -> GjkResult {
    let _eps = CsoPoint::FLT_EPS;
    let _eps_tol: f32 = CsoPoint::EPS_TOL;
    let _eps_rel: f32 = sqrt(_eps_tol);

    // TODO: reset the simplex if it is empty?
    var proj = VoronoiSimplex::project_origin_and_reduce(*simplex);

    var old_dir = Vector();
    let proj_len = length(proj.point);

    if proj_len != 0.0 {
        old_dir = -proj.point / proj_len;
    } else {
        return gjk_result_intersection();
    }

    var max_bound = 1.0e20;
    var dir = Vector();
    var niter = 0u;

    loop {
        let old_max_bound = max_bound;
        let proj_len = length(proj.point);

        if proj_len > CsoPoint::EPS_TOL {
            dir = -proj.point / proj_len;
            max_bound = proj_len;
        } else {
            // The origin is on the simplex.
            *simplex = proj.simplex;
            return gjk_result_intersection();
        }

        if max_bound >= old_max_bound {
            if exact_dist {
                let pts = result(proj.simplex, true);
                return gjk_result_closest_points(pts[0], pts[1], old_dir); // upper bounds inconsistencies
            } else {
                return gjk_result_proximity(old_dir);
            }
        }

        let cso_point = cso_point_from_shapes(pose12, g1, g2, dir);
        let min_bound = -dot(dir, cso_point.point);

        if min_bound > max_dist {
            return gjk_result_no_intersection(dir);
        } else if !exact_dist && min_bound > 0.0 && max_bound <= max_dist {
            return gjk_result_proximity(old_dir);
        } else if max_bound - min_bound <= _eps_rel * max_bound {
            if exact_dist {
                let pts = result(proj.simplex, false);
                return gjk_result_closest_points(pts[0], pts[1], dir); // the distance found has a good enough precision
            } else {
                return gjk_result_proximity(dir);
            }
        }

        let dim_before_add = proj.simplex.dim;
        proj.simplex = VoronoiSimplex::add_point(proj.simplex, cso_point);

        // Check if we pushed the same support point twice.
        if dim_before_add == proj.simplex.dim {
            if exact_dist {
                let pts = result(proj.simplex, false);
                return gjk_result_closest_points(pts[0], pts[1], dir);
            } else {
                return gjk_result_proximity(dir);
            }
        }

        old_dir = dir;
        proj = VoronoiSimplex::project_origin_and_reduce(proj.simplex);

        if proj.simplex.dim == DIM {
            if min_bound >= CsoPoint::EPS_TOL {
                if exact_dist {
                    let pts = result(proj.simplex, true);
                    return gjk_result_closest_points(pts[0], pts[1], old_dir);
                } else {
                    // NOTE: previous implementation used old_proj here.
                    return gjk_result_proximity(old_dir);
                }
            } else {
                *simplex = proj.simplex;
                return gjk_result_intersection(); // Point inside of the cso.
            }
        }
        niter += 1;

        if niter == 100 {
            break;
        }
    }

    return gjk_result_no_intersection(Vector(1.0, 0.0, 0.0));
}

fn result(simplex: VoronoiSimplex::VoronoiSimplex, prev: bool) -> array<Vector, 2> {
    var a = Vector();
    var b = Vector();

    if prev {
        for (var i = 0u; i < simplex.prev_dim + 1u; i++) {
            let coord = simplex.prev_proj[i];
            let point = simplex.vertices[simplex.prev_vertices[i]];
            a += point.orig_a * coord;
            b += point.orig_b * coord;
        }
    } else {
        for (var i = 0u; i < simplex.dim + 1u; i++) {
            let coord = simplex.proj[i];
            let point = simplex.vertices[i];
            a += point.orig_a * coord;
            b += point.orig_b * coord;
        }
    }

    return array(a, b);
}

/// Computes the support point of the CSO of `g1` and `g2` toward the direction `dir`.
fn cso_point_from_shapes(pos12: Transform, g1: ShapeA::Cuboid, g2: ShapeB::Cuboid, dir: Vector) -> CsoPoint::CsoPoint {
    let sp1 = ShapeA::local_support_point(g1, dir);
    let sp2 = ShapeB::support_point(g2, pos12, -dir);
    return CsoPoint::from_points(sp1, sp2);
}
