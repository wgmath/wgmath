#define_import_path wgparry::epa

#import wgebra::sim3 as Pose
#import wgparry::gjk::gjk as Gjk
#import wgparry::gjk::cso_point as CsoPoint
#import wgparry::gjk::voronoi_simplex as VoronoiSimplex
#import wgparry::triangle as Triangle
#import wgparry::projection as Proj
#import wgparry::cuboid as ShapeA
#import wgparry::cuboid as ShapeB

// TODO: find the ideal values.
const MAX_VERTICES_LEN: u32 = 32;
const MAX_FACES_LEN: u32 = 64;
const MAX_SILHOUETTE_LEN: u32 = 32;
const MAX_HEAP_LEN: u32 = 64;
const MAX_STACK_LEN: u32 = 32;

var<private> vertices: array<CsoPoint::CsoPoint, MAX_VERTICES_LEN>;
var<private> vertices_len: u32;
var<private> faces: array<Face, MAX_FACES_LEN>;
var<private> faces_len: u32;
var<private> silhouette: array<SilhouetteEdge, MAX_SILHOUETTE_LEN>;
var<private> silhouette_len: u32;
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
    pts: array<u32, 3>,
    adj: array<u32, 3>,
    normal: Vector,
    bcoords: vec3<f32>,
    deleted: bool,
}

struct FaceWithProj {
    face: Face,
    proj_inside: bool,
}

fn face_new_with_proj(
    bcoords: vec3<f32>,
    pts: array<u32, 3>,
    adj: array<u32, 3>,
) -> Face {
    let n = ccw_face_normal(
        vertices[pts[0]].point,
        vertices[pts[1]].point,
        vertices[pts[2]].point,
    );
    return Face(pts, adj, n, bcoords, false);
}

fn face_new(pts: array<u32, 3>, adj: array<u32, 3>) -> FaceWithProj {
    let tri = Triangle::Triangle(
        vertices[pts[0]].point,
        vertices[pts[1]].point,
        vertices[pts[2]].point,
    );
    let proj = Triangle::project_local_point_and_get_location(tri, Vector(), true);

    switch proj.feature_type {
        case Proj::FEATURE_VERTEX, Proj::FEATURE_EDGE: {
            let eps_tol = CsoPoint::FLT_EPS * 100.0; // Same as in closest_points
            let proj_inside = proj.inside || dot(proj.point, proj.point) < eps_tol * eps_tol;
            let bcoords = Proj::barycentric_coordinates(proj);
            return FaceWithProj(face_new_with_proj(bcoords, pts, adj), proj_inside);
        }
        case Proj::FEATURE_FACE: {
            return FaceWithProj(face_new_with_proj(proj.bcoords, pts, adj), true);
        }
        default: {
            return FaceWithProj(face_new_with_proj(vec3(), pts, adj), false);
        }
    }
}

fn face_closest_points(face: Face) -> array<vec3<f32>, 2> {
    return array(
        vertices[face.pts[0]].orig_a * face.bcoords[0]
            + vertices[face.pts[1]].orig_a * face.bcoords[1]
            + vertices[face.pts[2]].orig_a * face.bcoords[2],
        vertices[face.pts[0]].orig_b * face.bcoords[0]
            + vertices[face.pts[1]].orig_b * face.bcoords[1]
            + vertices[face.pts[2]].orig_b * face.bcoords[2],
    );
}

fn contains_point(face: Face, id: u32) -> bool {
    return face.pts[0] == id || face.pts[1] == id || face.pts[2] == id;
}

fn next_ccw_pt_id(face: Face, id: u32) -> u32 {
    if face.pts[0] == id {
        return 1;
    } else if face.pts[1] == id {
        return 2;
    } else {
        return 0;
    }
}

fn can_be_seen_by(face: Face, point: u32, opp_pt_id: u32) -> bool {
    let p0 = vertices[face.pts[opp_pt_id]].point;
    let p1 = vertices[face.pts[(opp_pt_id + 1) % 3]].point;
    let p2 = vertices[face.pts[(opp_pt_id + 2) % 3]].point;
    let pt = vertices[point].point;

    // NOTE: it is important that we return true for the case where
    // the dot product is zero. This is because degenerate faces will
    // have a zero normal, causing the dot product to be zero.
    // So return true for these case will let us skip the triangle
    // during silhouette computation.
    return dot(pt - p0, face.normal) >= -CsoPoint::EPS_TOL
        || Triangle::is_affinely_dependent(p1, p2, pt, CsoPoint::FLT_EPS * 100.0);
}

struct SilhouetteEdge {
    face_id: u32,
    opp_pt_id: u32,
}

//impl EPA {
//    fn project_origin<G: ?Sized + SupportMap>(
//        &mut self,
//        m: &Isometry<f32>,
//        g: &G,
//        simplex: &VoronoiSimplex,
//    ) -> Option<Point<f32>> {
//        self.closest_points(&m.inverse(), g, &ConstantOrigin, simplex)
//            .map(|(p, _, _)| p)
//    }

struct EpaResult {
    pt_a: vec3<f32>,
    pt_b: vec3<f32>,
    normal: vec3<f32>,
    valid: bool,
}

fn none() -> EpaResult {
    return EpaResult(vec3(), vec3(), vec3(), false);
}

fn closest_points(
    pos12: Transform,
    g1: ShapeA::Cuboid,
    g2: ShapeB::Cuboid,
    simplex: VoronoiSimplex::VoronoiSimplex,
) -> EpaResult {
    let _eps = CsoPoint::FLT_EPS;
    let _eps_tol = _eps * 100.0;

    /*
     * Reset buffers.
     */
   vertices_len = 0u;
   faces_len = 0u;
   silhouette_len = 0u;
   heap_len = 0u;

    /*
     * Initialization.
     */
    for (var i = 0u; i < simplex.dim + 1u; i++) {
        vertices[vertices_len] = simplex.vertices[i];
        vertices_len += 1u;
    }

    if simplex.dim == 0 {
        let n = vec3(0.0, 1.0, 0.0);
        return EpaResult(vec3(), vec3(), n, true);
    } else if simplex.dim == 3 {
        let dp1 = vertices[1].point - vertices[0].point;
        let dp2 = vertices[2].point - vertices[0].point;
        let dp3 = vertices[3].point - vertices[0].point;

        if dot(cross(dp1, dp2), dp3) > 0.0 {
            // Swap 1, 2
            let tmp = vertices[1];
            vertices[1] = vertices[2];
            vertices[2] = tmp;
        }

        let pts1 = array(0u, 1, 2);
        let pts2 = array(1u, 3, 2);
        let pts3 = array(0u, 2, 3);
        let pts4 = array(0u, 3, 1);

        let adj1 = array(3u, 1, 2);
        let adj2 = array(3u, 2, 0);
        let adj3 = array(0u, 1, 3);
        let adj4 = array(2u, 1, 0);

        let face1 = face_new(pts1, adj1);
        let face2 = face_new(pts2, adj2);
        let face3 = face_new(pts3, adj3);
        let face4 = face_new(pts4, adj4);

        faces[0] = face1.face;
        faces[1] = face2.face;
        faces[2] = face3.face;
        faces[3] = face4.face;
        faces_len = 4u;

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

        if face4.proj_inside {
            let dist4 = dot(faces[3].normal, vertices[3].point);
            let face_id = face_id_new(3, -dist4);

            if !face_id.valid {
                return none();
            }

            heap_push(face_id.face_id);
        }

        if !(face1.proj_inside || face2.proj_inside || face3.proj_inside || face4.proj_inside) {
            // Related issues:
            // https://github.com/dimforge/parry/issues/253
            // https://github.com/dimforge/parry/issues/246
            return none();
        }
    } else {
        if simplex.dim == 1 {
            let dpt = vertices[1].point - vertices[0].point;
            let basis = orthonormal_basis3(dpt);
            let cso_point_a = Gjk::cso_point_from_shapes(pos12, g1, g2, basis[0]);
            vertices[vertices_len] = cso_point_a;
            vertices_len += 1;
        }

        let pts1 = array(0u, 1, 2);
        let pts2 = array(0u, 2, 1);

        let adj1 = array(1u, 1, 1);
        let adj2 = array(0u, 0, 0);

        faces[0] = face_new(pts1, adj1).face;
        faces[1] = face_new(pts2, adj2).face;
        faces_len = 2;

        heap_push(face_id_new(0u, 0.0).face_id);
        heap_push(face_id_new(1u, 0.0).face_id);
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

        let cso_point = Gjk::cso_point_from_shapes(pos12, g1, g2, face.normal);
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

        faces[face_id.id].deleted = true;

        let adj_opp_pt_id1 = next_ccw_pt_id(faces[face.adj[0]], face.pts[0]);
        let adj_opp_pt_id2 = next_ccw_pt_id(faces[face.adj[1]], face.pts[1]);
        let adj_opp_pt_id3 = next_ccw_pt_id(faces[face.adj[2]], face.pts[2]);

        if !compute_silhouette(support_point_id, face.adj[0], adj_opp_pt_id1) {
            return none(); // out of memory
        }
        if !compute_silhouette(support_point_id, face.adj[1], adj_opp_pt_id2) {
            return none(); // out of memory
        }
        if !compute_silhouette(support_point_id, face.adj[2], adj_opp_pt_id3) {
            return none(); // out of memory
        }

        let first_new_face_id = faces_len;

        if silhouette_len == 0 {
            // TODO: Something went very wrong because we failed to extract a silhouette…
            return none();
        }

        for (var eid = 0u; eid < silhouette_len; eid++) {
            let edge = silhouette[eid];
            if !faces[edge.face_id].deleted {
                let new_face_id = faces_len;

                let pt_id1 = faces[edge.face_id].pts[(edge.opp_pt_id + 2) % 3];
                let pt_id2 = faces[edge.face_id].pts[(edge.opp_pt_id + 1) % 3];

                let pts = array(pt_id1, pt_id2, support_point_id);
                let adj = array(edge.face_id, new_face_id + 1, new_face_id - 1);
                let new_face = face_new(pts, adj);

                faces[edge.face_id].adj[(edge.opp_pt_id + 1) % 3] = new_face_id;

                if faces_len != MAX_FACES_LEN {
                    faces[faces_len] = new_face.face;
                    faces_len += 1;
                } else {
                    return none();
                }

                if new_face.proj_inside {
                    let pt = vertices[faces[new_face_id].pts[0]].point;
                    let dist = dot(faces[new_face_id].normal, pt);
                    if dist < curr_dist {
                        // TODO: if we reach this point, there were issues due to
                        // numerical errors.
                        let points = face_closest_points(face);
                        return EpaResult(points[0], points[1], face.normal, true);
                    }

                    let to_push = face_id_new(new_face_id, -dist);

                    if !to_push.valid {
                        return none();
                    }

                    if !heap_push(to_push.face_id) {
                        return none();
                    }
                }
            }
        }

        if first_new_face_id == faces_len {
            // Something went very wrong because all the edges
            // from the silhouette belonged to deleted faces.
            return none();
        }

        faces[first_new_face_id].adj[2] = faces_len - 1;
        faces[faces_len - 1].adj[1] = first_new_face_id;

        // Clear silhouette buffer.
        silhouette_len = 0;

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

// Returns `false` if running out of memory.
fn compute_silhouette(point: u32, id: u32, opp_pt_id: u32) -> bool {
    var stack = array<SilhouetteEdge, MAX_STACK_LEN>();
    var stack_len = 1u;
    stack[0u] = SilhouetteEdge(id, opp_pt_id);

    while stack_len > 0u {
        stack_len -= 1u;
        let edge = stack[stack_len];

        if !faces[edge.face_id].deleted {
            if !can_be_seen_by(faces[edge.face_id], point, edge.opp_pt_id) {
                if silhouette_len < MAX_STACK_LEN {
                    silhouette[silhouette_len] = SilhouetteEdge(edge.face_id, edge.opp_pt_id);
                    silhouette_len += 1u;
                } else {
                    return false;
                }
            } else {
                faces[edge.face_id].deleted = true;

                let adj_pt_id1 = (edge.opp_pt_id + 2) % 3;
                let adj_pt_id2 = edge.opp_pt_id;

                let adj1 = faces[edge.face_id].adj[adj_pt_id1];
                let adj2 = faces[edge.face_id].adj[adj_pt_id2];

                let adj_opp_pt_id1 =
                    next_ccw_pt_id(faces[adj1], faces[edge.face_id].pts[adj_pt_id1]);
                let adj_opp_pt_id2 =
                    next_ccw_pt_id(faces[adj2], faces[edge.face_id].pts[adj_pt_id2]);

                stack[stack_len] = SilhouetteEdge(adj2, adj_opp_pt_id2);
                stack_len += 1u;

                if stack_len < MAX_STACK_LEN {
                    stack[stack_len] = SilhouetteEdge(adj1, adj_opp_pt_id1);
                    stack_len += 1u;
                } else {
                    return false;
                }
            }
        }
    }

    return true;
}

fn ccw_face_normal(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let res = cross(ab, ac);
    let res_length = length(res);
    return select(vec3(), res / res_length, res_length > CsoPoint::FLT_EPS);
}

// TODO: refactor to its own file. We already have a copy of this function in capsule.wgsl
fn orthonormal_basis3(v: vec3<f32>) -> array<vec3<f32>, 2> {
    // NOTE: not using `sign` because we don’t want the 0.0 case to return 0.0.
    let sign = select(-1.0, 1.0, v.z >= 0.0);
    let a = -1.0 / (sign + v.z);
    let b = v.x * v.y * a;

    return array(
        vec3(
            1.0 + sign * v.x * v.x * a,
            sign * b,
            -sign * v.x,
        ),
        vec3(b, sign + v.y * v.y * a, -v.y),
    );
}