#define_import_path wgparry::epa

#import wgebra::sim2 as Pose
#import wgparry::gjk::gjk as Gjk
#import wgparry::gjk::cso_point as CsoPoint
#import wgparry::gjk::voronoi_simplex as VoronoiSimplex
#import wgparry::triangle as Triangle
#import wgparry::projection as Proj
#import wgparry::shape as Shape


// TODO: find the ideal values.
const MAX_VERTICES_LEN: u32 = 32;
const MAX_FACES_LEN: u32 = 32;
const MAX_HEAP_LEN: u32 = 32;

var<private> vertices: array<CsoPoint::CsoPoint, MAX_VERTICES_LEN>;
var<private> vertices_len: u32;
var<private> faces: array<Face, MAX_FACES_LEN>;
var<private> faces_len: u32;
var<private> heap: array<FaceId, MAX_HEAP_LEN>; // TODO: BinaryHeap<FaceId>
var<private> heap_len: u32;

fn heap_best_index() -> u32 {
    var best_id = 0u;

    for (var i = 0u; i < heap_len; i++) {
        if heap[i].neg_dist > heap[best_id].neg_dist {
            best_id = i;
        }
    }

    return best_id;
}

fn heap_peek() -> FaceId {
    return heap[heap_best_index()];
}

fn heap_pop() -> FaceId {
    // TODO PERF: implement an actual priority queue.
    let i = heap_best_index();
    let result = heap[i];

    if heap_len != 0u {
        heap_len -= 1u;
        heap[i] = heap[heap_len]; // Swap-remove.
    } else {
        /* unreachable */
    }

    return result;
}

// If it runs out of memory, returns `false`.
fn heap_push(elt: FaceId) -> bool {
    if heap_len != MAX_HEAP_LEN {
        heap[heap_len] = elt;
        heap_len += 1;
        return true;
    } else {
        return false;
    }
}

struct FaceId {
    id: u32,
    neg_dist: f32,
}

struct OptionFaceId {
    face_id: FaceId,
    valid: bool,
}

fn face_id_new(id: u32, neg_dist: f32) -> OptionFaceId {
    if neg_dist > CsoPoint::EPS_TOL {
        return OptionFaceId(FaceId(), false);
    } else {
        return OptionFaceId(FaceId(id, neg_dist), true);
    }
}

struct Face {
    pts: array<u32, 2>,
    normal: Vector,
    proj: vec2<f32>,
    bcoords: vec2<f32>,
    deleted: bool,
}

struct FaceWithProj {
    face: Face,
    proj_inside: bool,
}

fn face_new(pts: array<u32, 2>) -> FaceWithProj {
    let proj = project_origin(vertices[pts[0]].point, vertices[pts[1]].point);
    if proj.valid {
        return FaceWithProj(face_new_with_proj(proj.point, proj.bcoords, pts), true);
    } else {
        return FaceWithProj(face_new_with_proj(vec2(), vec2(), pts), false);
    }
}

fn face_new_with_proj(
    proj: vec2<f32>,
    bcoords: vec2<f32>,
    pts: array<u32, 2>,
) -> Face {
    let n = ccw_face_normal(
        vertices[pts[0]].point,
        vertices[pts[1]].point,
    );
    return Face(pts, n, proj, bcoords, all(n == vec2()));
}

fn face_closest_points(face: Face) -> array<vec2<f32>, 2> {
    return array(
        vertices[face.pts[0]].orig_a * face.bcoords[0]
            + vertices[face.pts[1]].orig_a * face.bcoords[1],
        vertices[face.pts[0]].orig_b * face.bcoords[0]
            + vertices[face.pts[1]].orig_b * face.bcoords[1],
    );
}

struct EpaResult {
    pt_a: vec2<f32>,
    pt_b: vec2<f32>,
    normal: vec2<f32>,
    valid: bool,
}

fn none() -> EpaResult {
    return EpaResult(vec2(), vec2(), vec2(), false);
}

fn closest_points(
    pose12: Transform,
    g1: Shape::Shape,
    g2: Shape::Shape,
    simplex: VoronoiSimplex::VoronoiSimplex,
) -> EpaResult
{
    let _eps = CsoPoint::FLT_EPS;
    let _eps_tol = _eps * 100.0;

    /*
     * Reset buffers.
     */
   vertices_len = 0u;
   faces_len = 0u;
   heap_len = 0u;

    /*
     * Initialization.
     */
    for (var i = 0u; i < simplex.dim + 1u; i++) {
        vertices[vertices_len] = simplex.vertices[i];
        vertices_len += 1u;
    }

    if simplex.dim == 0 {
        const MAX_ITERS: u32 = 100; // If there is no convergence, just use whatever direction was extracted so far

        // The contact is vertex-vertex.
        // We need to determine a valid normal that lies
        // on both vertices' normal cone.
        var n = vec2(0.0, 1.0);

        // First, find a vector on the first vertex tangent cone.
        let orig1 = vertices[0].orig_a;
        for (var k = 0u; k < MAX_ITERS; k++) {
            let supp1 = Shape::local_support_point(g1, n);
            var tangent = supp1 - orig1;
            let tangent_len = length(tangent);

            if tangent_len > _eps_tol {
                tangent /= tangent_len;
                if dot(n, tangent) < _eps_tol {
                    break;
                }

                n = vec2(-tangent.y, tangent.x);
            } else {
                break;
            }
        }

        // Second, ensure the direction lies on the second vertex's tangent cone.
        let orig2 = vertices[0].orig_b;
        for (var k = 0u; k < MAX_ITERS; k++) {
            let supp2 = Shape::support_point(g2, pose12, -n);
            var tangent = supp2 - orig2;
            let tangent_len = length(tangent);

            if tangent_len > _eps_tol {
                tangent /= tangent_len;
                if dot(-n, tangent) < _eps_tol {
                    break;
                }

                n = vec2(-tangent.y, tangent.x);
            } else {
                break;
            }
        }

        return EpaResult(vec2(), vec2(), n, true);
    } else if simplex.dim == 2 {
        let dp1 = vertices[1].point - vertices[0].point;
        let dp2 = vertices[2].point - vertices[0].point;

        if Triangle::perp(dp1, dp2) < 0.0 {
            let vtx1 = vertices[1];
            vertices[1] = vertices[2];
            vertices[2] = vtx1;
        }

        let pts1 = array(0u, 1);
        let pts2 = array(1u, 2);
        let pts3 = array(2u, 0);

        let face1 = face_new(pts1);
        let face2 = face_new(pts2);
        let face3 = face_new(pts3);

        faces[0] = face1.face;
        faces[1] = face2.face;
        faces[2] = face3.face;
        faces_len = 3;

        if face1.proj_inside {
            let dist1 = dot(faces[0].normal, vertices[0].point);
            let face_id = face_id_new(0, -dist1);

            if !face_id.valid {
                return none();
            }

            heap_push(face_id.face_id);
        }

        if face2.proj_inside {
            let dist2 = dot(faces[1].normal, vertices[1].point);
            let face_id = face_id_new(1, -dist2);

            if !face_id.valid {
                return none();
            }

            heap_push(face_id.face_id);
        }

        if face3.proj_inside {
            let dist3 = dot(faces[2].normal, vertices[2].point);
            let face_id = face_id_new(2, -dist3);

            if !face_id.valid {
                return none();
            }

            heap_push(face_id.face_id);
        }

        if !(face1.proj_inside || face2.proj_inside || face3.proj_inside) {
            // Related issues:
            // https://github.com/dimforge/parry/issues/253
            // https://github.com/dimforge/parry/issues/246
            return none();
        }
    } else {
        let pts1 = array(0u, 1);
        let pts2 = array(1u, 0);

        faces[0] = face_new_with_proj(vec2(), vec2(1.0, 0.0), pts1);
        faces[1] = face_new_with_proj(vec2(), vec2(1.0, 0.0), pts2);
        faces_len = 2;

        let dist1 = dot(faces[0].normal, vertices[0].point);
        let dist2 = dot(faces[1].normal, vertices[1].point);
        let fid1 = face_id_new(0u, dist1);
        let fid2 = face_id_new(1u, dist2);

        if !fid1.valid {
            return none();
        }
        if !fid2.valid {
            return none();
        }

        heap_push(fid1.face_id);
        heap_push(fid2.face_id);
    }

    if heap_len == 0 {
        return none();
    }

    var niter = 0u;
    var max_dist = 1.0e20;
    var best_face_id = heap_peek();
    var old_dist = 0.0;

    /*
     * Run the expansion.
     */
    while heap_len > 0u {
        let face_id = heap_pop();
        // Create new faces.
        let face = faces[face_id.id];

        if face.deleted {
            continue;
        }

        let cso_point = Gjk::cso_point_from_shapes(pose12, g1, g2, face.normal);
        let support_point_id = vertices_len;

        if vertices_len != MAX_VERTICES_LEN {
            vertices[vertices_len] = cso_point;
            vertices_len += 1;
        } else {
            // We ran out of memory.
            return none();
        }

        let candidate_max_dist = dot(cso_point.point, face.normal);

        if candidate_max_dist < max_dist {
            best_face_id = face_id;
            max_dist = candidate_max_dist;
        }

        let curr_dist = -face_id.neg_dist;

        if max_dist - curr_dist < _eps_tol ||
            // Accept the intersection as the algorithm is stuck and no new points will be found
            // This happens because of numerical stability issue
            (abs(curr_dist - old_dist) < _eps && candidate_max_dist < max_dist)
        {
            let best_face = faces[best_face_id.id];
            let points = face_closest_points(best_face);
            return EpaResult(points[0], points[1], best_face.normal, true);
        }

        old_dist = curr_dist;

        let pts1 = array(face.pts[0], support_point_id);
        let pts2 = array(support_point_id, face.pts[1]);

        let new_faces = array(
            face_new(pts1),
            face_new(pts2),
        );

        // TODO: unroll
        for (var fi = 0u; fi < 2u; fi++) {
            let f = new_faces[fi];
            if f.proj_inside {
                let dist = dot(f.face.normal, f.face.proj);
                if dist < curr_dist {
                    // TODO: if we reach this point, there were issues due to
                    // numerical errors.
                    let cpts = face_closest_points(f.face);
                    return EpaResult(cpts[0], cpts[1], f.face.normal, true);
                }

                if !f.face.deleted {
                    let new_fid = face_id_new(faces_len, -dist);
                    if !new_fid.valid {
                        return none();
                    }

                    if !heap_push(new_fid.face_id) {
                        return none();
                    }
                }
            }

            if faces_len != MAX_FACES_LEN {
                faces[faces_len] = f.face;
                faces_len += 1;
            } else {
                return none();
            }
        }

        niter += 1;
        if niter > 100 {
            // if we reached this point, our algorithm didn't converge to what precision we wanted.
            // still return an intersection point, as it's probably close enough.
            break;
        }
    }

    let best_face = faces[best_face_id.id];
    let points = face_closest_points(best_face);
    return EpaResult(points[0], points[1], best_face.normal, true);
}

struct ProjectOriginResult {
    point: vec2<f32>,
    bcoords: vec2<f32>,
    valid: bool,
}

fn project_origin(a: vec2<f32>, b: vec2<f32>) -> ProjectOriginResult {
    let ab = b - a;
    let ap = -a;
    let ab_ap = dot(ab, ap);
    let sqnab = dot(ab, ab);

    if sqnab == 0.0 {
        return ProjectOriginResult(vec2(), vec2(), false);
    }

    if ab_ap < -CsoPoint::EPS_TOL || ab_ap > sqnab + CsoPoint::EPS_TOL {
        // Voronoï region of vertex 'a' or 'b'.
        return ProjectOriginResult(vec2(), vec2(), false);
    } else {
        // Voronoï region of the segment interior.
        let position_on_segment = ab_ap / sqnab;
        let res = a + ab * position_on_segment;

        return ProjectOriginResult(res, vec2(1.0 - position_on_segment, position_on_segment), true);
    }
}

fn ccw_face_normal(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let ab = b - a;
    let res = vec2(ab.y, -ab.x);
    let res_length = length(res);
    return select(vec2(), res / res_length, res_length > CsoPoint::FLT_EPS);
}