//! Polygonal Feature Contact Generation
//!
//! This module implements contact point generation between polygonal features of convex shapes.
//! After SAT identifies a separating axis, this module clips the support features to generate
//! a complete contact manifolds immediately.
//!
//! Key concepts:
//! - PolygonalFeature: Represents a face, edge, or vertex as a polygon with up to 4 vertices (2 in 2D)
//! - Clipping: polygon-polygon clipping projected onto the 2D contact plane
//! - Manifold Reduction: Reduces potentially many contact candidates to the most important 4 (or 2 in 2D)

#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif

#import wgparry::contact_manifold as Manifold

#define_import_path wgparry::polygonal_feature

// TODO: share the epsilon value across modules?
const EPSILON: f32 = 1.1920929E-7;
/// Cosine of pi/8 (approximately 22.5 degrees), used for parallelism tests.
const COS_FRAC_PI_8: f32 = 0.92387953251;
/// Maximum floating point value (approximation for sentinel values).
const MAX_FLT: f32 = 3.4e38; // TODO: the actual exact value isn't accepted by the browser: 3.40282347E+38;

#if DIM == 2
/// Maximum vertices in a 2D polygonal feature (edge).
const MAX_VERTICES: u32 = 2;
#else
/// Maximum vertices in a 3D polygonal feature (quad face).
const MAX_VERTICES: u32 = 4;
#endif

/// A polygonal feature representing the local polygonal approximation of
/// a vertex, face, or edge of a convex shape.
///
/// This can represent:
/// - A vertex (num_vertices = 1)
/// - An edge (num_vertices = 2)
/// - A face (num_vertices = 3 or 4 in 3D)
struct PolygonalFeature {
    /// Up to four vertices forming this polygonal feature.
    vertices: array<Vector, MAX_VERTICES>,
    /// The number of vertices in this feature.
    num_vertices: u32,
}

/// Transform each vertex of this polygonal feature by the given pose.
fn transform_by(poly: PolygonalFeature, pose: Transform) -> PolygonalFeature {
    return PolygonalFeature(
        array(
            Pose::mulPt(pose, poly.vertices[0]),
            Pose::mulPt(pose, poly.vertices[1]),
#if DIM == 3
            Pose::mulPt(pose, poly.vertices[2]),
            Pose::mulPt(pose, poly.vertices[3]),
#endif
        ),
        poly.num_vertices
    );
}

#if DIM == 2
/// Computes the contacts between two polygonal features.
fn contacts(
    pose12: Pose::Sim2,
    pose21: Pose::Sim2,
    sep_axis1: vec2<f32>,
    sep_axis2: vec2<f32>,
    feature1: PolygonalFeature,
    feature2: PolygonalFeature,
    prediction: f32,
    flipped: bool,
) -> Manifold::ContactManifold {
    if feature1.num_vertices == 2 {
        if feature2.num_vertices == 2 {
            return face_face_contacts(pose12, feature1, sep_axis1, feature2, prediction, flipped);
        } else {
            return face_vertex_contacts(pose12, feature1, sep_axis1, feature2, prediction, flipped);
        }
    } else {
        return face_vertex_contacts(pose21, feature2, sep_axis2, feature1, prediction, !flipped);
    }
}

/// Compute contacts points between a face and a vertex.
///
/// This method assume we already know that at least one contact exists.
fn face_vertex_contacts(
    pose12: Pose::Sim2,
    face1: PolygonalFeature,
    sep_axis1: vec2<f32>,
    vertex2: PolygonalFeature,
    prediction: f32,
    flipped: bool,
) -> Manifold::ContactManifold {
    var result = Manifold::ContactManifold();
    let v2_1 = Pose::mulPt(pose12, vertex2.vertices[0]);
    let tangent1 = face1.vertices[1] - face1.vertices[0];
    let normal1 = Vector(-tangent1.y, tangent1.x);
    let denom = -dot(normal1, sep_axis1);
    let dist = dot(face1.vertices[0] - v2_1, normal1) / denom;

    if dist < prediction {
        let local_p2 = v2_1;
        let local_p1 = v2_1 - dist * normal1;

        if !flipped {
            result.points_a[0] = Manifold::ContactPoint(local_p1, dist);
        } else {
            let local_p2 = Pose::invMulPt(pose12, v2_1);
            result.points_a[0] = Manifold::ContactPoint(local_p2, dist);
        }
        result.len = 1;
    }

    return result;
}

/// Computes the contacts between two polygonal faces.
fn face_face_contacts(
    pose12: Pose::Sim2,
    face1: PolygonalFeature,
    normal1: vec2<f32>,
    face2: PolygonalFeature,
    prediction: f32,
    flipped: bool,
) -> Manifold::ContactManifold {
    var result = Manifold::ContactManifold();

    let clip = clip_segment_segment_with_normal(
        face1.vertices[0],
        face1.vertices[1],
        Pose::mulPt(pose12, face2.vertices[0]),
        Pose::mulPt(pose12, face2.vertices[1]),
        normal1,
    );

    if !clip.empty {
        let dist_a = dot(clip.seg2_a - clip.seg1_a, normal1);

        if dist_a < prediction {
            if !flipped {
                result.points_a[0] = Manifold::ContactPoint(clip.seg1_a, dist_a);
            } else {
                let local_p2 = Pose::invMulPt(pose12, clip.seg2_a);
                result.points_a[0] = Manifold::ContactPoint(local_p2, dist_a);
            }
            result.len = 1;
        }

        let dist_b = dot(clip.seg2_b - clip.seg1_b, normal1);
        if dist_b < prediction {
            let i = result.len;
            if !flipped {
                result.points_a[i] = Manifold::ContactPoint(clip.seg1_b, dist_b);
            } else {
                let local_p2 = Pose::invMulPt(pose12, clip.seg2_b);
                result.points_a[i] = Manifold::ContactPoint(local_p2, dist_b);
            }
            result.len += 1;
        }
    }

    return result;
}

struct ClippingPoints {
    seg1_a: vec2<f32>,
    seg2_a: vec2<f32>,
    seg1_b: vec2<f32>,
    seg2_b: vec2<f32>,
    empty: bool,
}

fn clip_segment_segment_with_normal(
    seg1_a_: vec2<f32>,
    seg1_b_: vec2<f32>,
    seg2_a_: vec2<f32>,
    seg2_b_: vec2<f32>,
    normal: vec2<f32>,
) -> ClippingPoints {
    var seg1_a = seg1_a_;
    var seg1_b = seg1_b_;
    var seg2_a = seg2_a_;
    var seg2_b = seg2_b_;

    let tangent = vec2(-normal.y, normal.x);
    var result = ClippingPoints();
    var range1 = array(dot(seg1_a, tangent), dot(seg1_b, tangent));
    var range2 = array(dot(seg2_a, tangent), dot(seg2_b, tangent));

    if range1[1] < range1[0] {
        var swp = seg1_a;
        seg1_a = seg1_b;
        seg1_b = swp;
        range1 = array(range1[1], range1[0]);
    }

    if range2[1] < range2[0] {
        var swp = seg2_a;
        seg2_a = seg2_b;
        seg2_b = swp;
        range2 = array(range2[1], range2[0]);
    }

    if range2[0] > range1[1] || range1[0] > range2[1] {
        // No clip point.
        result.empty = true;
        return result;
    }

    if range2[0] > range1[0] {
        let bcoord = (range2[0] - range1[0]) * pseudo_inv(range1[1] - range1[0]);
        result.seg1_a = seg1_a + (seg1_b - seg1_a) * bcoord;
        result.seg2_a = seg2_a;
    } else {
        let bcoord = (range1[0] - range2[0]) * pseudo_inv(range2[1] - range2[0]);
        result.seg1_a = seg1_a;
        result.seg2_a = seg2_a + (seg2_b - seg2_a) * bcoord;
    }

    if range2[1] < range1[1] {
        let bcoord = (range2[1] - range1[0]) * pseudo_inv(range1[1] - range1[0]);
        result.seg1_b = seg1_a + (seg1_b - seg1_a) * bcoord;
        result.seg2_b = seg2_b;
    } else {
        let bcoord = (range1[1] - range2[0]) * pseudo_inv(range2[1] - range2[0]);
        result.seg1_b = seg1_b;
        result.seg2_b = seg2_a + (seg2_b - seg2_a) * bcoord;
    }

    return result;
}

fn pseudo_inv(x: f32) -> f32 {
    return select(1.0 / x, 0.0, x == 0.0);
}

#else
/* DIM3 */

/// Computes all the contacts between two polygonal features.
fn contacts(
    pose12: Pose::Sim3,
    _pos21: Pose::Sim3, // Unused argument, to match the 2D definition.
    sep_axis1: vec3<f32>,
    sep_axis2: vec3<f32>,
    feature1: PolygonalFeature,
    feature2: PolygonalFeature,
    prediction: f32,
    flipped: bool,
) -> Manifold::ContactManifold {
    if feature1.num_vertices == 2 && feature2.num_vertices == 2 {
        return contacts_edge_edge(pose12, feature1, sep_axis1, feature2, prediction, flipped);
    } else {
        return contacts_face_face(pose12, feature1, sep_axis1, feature2, prediction, flipped);
    }
}

fn contacts_edge_edge(
    pose12: Pose::Sim3,
    face1: PolygonalFeature,
    sep_axis1: vec3<f32>,
    face2: PolygonalFeature,
    prediction: f32,
    flipped: bool,
) -> Manifold::ContactManifold {
    // Project the faces to a 2D plane for contact clipping.
    // The plane they are projected onto has normal sep_axis1
    // and contains the origin (this is numerically OK because
    // we are not working in world-space here).
    var result = Manifold::ContactManifold();
    var basis = orthonormal_basis3(sep_axis1);
    var projected_edge1 = array(
        vec2(
            dot(face1.vertices[0], basis[0]),
            dot(face1.vertices[0], basis[1]),
        ),
        vec2(
            dot(face1.vertices[1], basis[0]),
            dot(face1.vertices[1], basis[1]),
        ),
    );

    var vertices2_1 = array(Pose::mulPt(pose12, face2.vertices[0]), Pose::mulPt(pose12, face2.vertices[1]));
    var projected_edge2 = array(
        vec2(
            dot(vertices2_1[0], basis[0]),
            dot(vertices2_1[0], basis[1]),
        ),
        vec2(
            dot(vertices2_1[1], basis[0]),
            dot(vertices2_1[1], basis[1]),
        ),
    );

    var tangent1 = projected_edge1[1] - projected_edge1[0];
    var tangent2 = projected_edge2[1] - projected_edge2[0];
    let tangent_len1 = length(tangent1);
    let tangent_len2 = length(tangent2);

    // TODO: not sure what the best value for eps is.
    if tangent_len1 > EPSILON && tangent_len2 > EPSILON {
        tangent1 /= tangent_len1;
        tangent2 /= tangent_len2;

        let parallel = dot(tangent1, tangent2) >= COS_FRAC_PI_8;

        if !parallel {
            let bcoords = closest_points_segment_segment(
                projected_edge1[0],
                projected_edge1[1],
                projected_edge2[0],
                projected_edge2[1],
            );

            // Found a contact between the two edges.
            let local_p1 = face1.vertices[0] * (1.0 - bcoords.x) + face1.vertices[1] * bcoords.x;
            let local_p2_1 = vertices2_1[0] * (1.0 - bcoords.y) + vertices2_1[1] * bcoords.y;
            let dist = dot(local_p2_1 - local_p1, sep_axis1);

            if dist <= prediction {
                if !flipped {
                    result.points_a[0] = Manifold::ContactPoint(local_p1, dist);
                } else {
                    let local_p2 = Pose::invMulPt(pose12, local_p2_1);
                    result.points_a[0] = Manifold::ContactPoint(local_p2, dist);
                }
                result.len = 1;
            }
            return result;
        }
    }

    // The lines are parallel so we are having a conformal contact.
    // Let's use a range-based clipping to extract two contact points.
    // TODO: would it be better and/or more efficient to do the
    //clipping in 2D?
    let clips = clip_segment_segment(
        face1.vertices[0],
        face1.vertices[1],
        vertices2_1[0],
        vertices2_1[1],
    );

    if !clips.empty {
        let dist0 = dot(clips.seg2_a - clips.seg1_a, sep_axis1);
        let dist1 = dot(clips.seg2_b - clips.seg1_b, sep_axis1);

        if dist0 <= prediction {
            if !flipped {
                result.points_a[0] = Manifold::ContactPoint(clips.seg1_a, dist0);
            } else {
                let local_p2 = Pose::invMulPt(pose12, clips.seg2_a);
                result.points_a[0] = Manifold::ContactPoint(local_p2, dist0);
            }

            result.len = 1;
        }

        let k = result.len;

        if dist1 <= prediction {
            if !flipped {
                result.points_a[k] = Manifold::ContactPoint(clips.seg1_b, dist1);
            } else {
                let local_p2 = Pose::invMulPt(pose12, clips.seg2_b);
                result.points_a[k] = Manifold::ContactPoint(local_p2, dist1);
            }
            result.len += 1;
        }
    }

    return result;
}

const MAX_CANDIDATE_POINTS: u32 = 8;

fn contacts_face_face(
    pose12: Pose::Sim3,
    face1: PolygonalFeature,
    sep_axis1: vec3<f32>,
    face2: PolygonalFeature,
    prediction: f32,
    flipped: bool,
) -> Manifold::ContactManifold {
    // Project the faces to a 2D plane for contact clipping.
    // The plane they are projected onto has normal sep_axis1
    // and contains the origin (this is numerically OK because
    // we are not working in world-space here).
    var candidates: array<Manifold::ContactPoint, MAX_CANDIDATE_POINTS> = array(
        Manifold::ContactPoint(),
        Manifold::ContactPoint(),
        Manifold::ContactPoint(),
        Manifold::ContactPoint(),
        Manifold::ContactPoint(),
        Manifold::ContactPoint(),
        Manifold::ContactPoint(),
        Manifold::ContactPoint(),
    );
    var num_candidates = 0u;

    var basis = orthonormal_basis3(sep_axis1);
    var projected_face1 = array(
        vec2(
            dot(face1.vertices[0], basis[0]),
            dot(face1.vertices[0], basis[1]),
        ),
        vec2(
            dot(face1.vertices[1], basis[0]),
            dot(face1.vertices[1], basis[1]),
        ),
        vec2(
            dot(face1.vertices[2], basis[0]),
            dot(face1.vertices[2], basis[1]),
        ),
        vec2(
            dot(face1.vertices[3], basis[0]),
            dot(face1.vertices[3], basis[1]),
        ),
    );

    var vertices2_1 = array(
        Pose::mulPt(pose12, face2.vertices[0]),
        Pose::mulPt(pose12, face2.vertices[1]),
        Pose::mulPt(pose12, face2.vertices[2]),
        Pose::mulPt(pose12, face2.vertices[3]),
    );
    var projected_face2 = array(
        vec2(
            dot(vertices2_1[0], basis[0]),
            dot(vertices2_1[0], basis[1]),
        ),
        vec2(
            dot(vertices2_1[1], basis[0]),
            dot(vertices2_1[1], basis[1]),
        ),
        vec2(
            dot(vertices2_1[2], basis[0]),
            dot(vertices2_1[2], basis[1]),
        ),
        vec2(
            dot(vertices2_1[3], basis[0]),
            dot(vertices2_1[3], basis[1]),
        ),
    );

    // Also find all the vertices located inside of the other projected face.
    if face2.num_vertices > 2u {
        let normal2_1 =
            cross(vertices2_1[2] - vertices2_1[1], vertices2_1[0] - vertices2_1[1]);
        let denom = dot(normal2_1, sep_axis1);

        if !relative_eq(denom, 0.0) {
            let last_index2 = face2.num_vertices - 1;
            var any_point_is_outside = false;

            for (var i = 0u; i < face1.num_vertices; i++) {
                let p1 = projected_face1[i];

                var sign = perp(projected_face2[0] - projected_face2[last_index2], p1 - projected_face2[last_index2]);

                var point_is_outside = false;
                for (var j = 0u; j < last_index2; j++) {
                    let new_sign = perp(projected_face2[j + 1] - projected_face2[j], p1 - projected_face2[j]);

                    if sign == 0.0 {
                        sign = new_sign;
                    } else if sign * new_sign < 0.0 {
                        // The point lies outside.
                        point_is_outside = true;
                        break;
                    }
                }

                any_point_is_outside = any_point_is_outside || point_is_outside;
                let dist = dot(vertices2_1[0] - face1.vertices[i], normal2_1) / denom;

                if !point_is_outside && dist <= prediction {
                    // All the perp had the same sign: the point is inside of the other shapes projection.
                    // Output the contact.
                    let local_p1 = face1.vertices[i];
                    let local_p2_1 = face1.vertices[i] + dist * sep_axis1;

                    if dist <= prediction {
                        if !flipped {
                            // NOTE: for some unknown reason, doing: `candidates[num_candidates] = ContactPoint(local_p1, dist)`
                            //       results in incorrect data when running on my rtx 5080 laptop (but is OK on the
                            //       integrated gpus).
                            candidates[num_candidates].pt = local_p1;
                            candidates[num_candidates].dist = dist;
                        } else {
                            let local_p2 = Pose::invMulPt(pose12, local_p2_1);
                            candidates[num_candidates].pt = local_p2;
                            candidates[num_candidates].dist = dist;
                        }
                        num_candidates += 1u;
                    }
                }
            }

            if !any_point_is_outside {
                // If all the vertices of face1 projected on face2, we can early-exist
                // since convexity guarantees face1 is fully inside face2.
                return manifold_reduction(candidates, num_candidates, sep_axis1);
            }
        }
    }

    if face1.num_vertices > 2u {
        let normal1 = cross(face1.vertices[2] - face1.vertices[1], face1.vertices[0] - face1.vertices[1]);

        let denom = -dot(normal1, sep_axis1);
        if !relative_eq(denom, 0.0) {
            let last_index1 = face1.num_vertices - 1u;
            var any_point_is_outside = false;

            for (var i = 0u; i < face2.num_vertices; i++) {
                let p2 = projected_face2[i];

                var sign = perp(projected_face1[0] - projected_face1[last_index1], p2 - projected_face1[last_index1]);

                var point_is_outside = false;
                for (var j = 0u; j < last_index1; j++) {
                    let new_sign = perp(projected_face1[j + 1u] - projected_face1[j], p2 - projected_face1[j]);

                    if sign == 0.0 {
                        sign = new_sign;
                    } else if sign * new_sign < 0.0 {
                        // The point lies outside.
                        point_is_outside = true;
                        break;
                    }
                }

                any_point_is_outside = any_point_is_outside || point_is_outside;
                let dist = dot(face1.vertices[0] - vertices2_1[i], normal1) / denom;

                if !point_is_outside && dist <= prediction {
                    // All the perp had the same sign: the point is inside of the other shapes projection.
                    // Output the contact.
                    let local_p2_1 = vertices2_1[i];
                    let local_p1 = vertices2_1[i] - dist * sep_axis1;

                    if dist <= prediction {
                        if !flipped {
                            candidates[num_candidates].pt = local_p1;
                            candidates[num_candidates].dist = dist;
                        } else {
                            let local_p2 = Pose::invMulPt(pose12, local_p2_1);
                            candidates[num_candidates].pt = local_p2;
                            candidates[num_candidates].dist = dist;
                        }
                        num_candidates += 1u;
                    }
                }
            }

            if !any_point_is_outside {
                // If all the vertices of face1 projected on face2, we can early-exist
                // since convexity guarantees face1 is fully inside face2.
                return manifold_reduction(candidates, num_candidates, sep_axis1);
            }
        }
    }

    // Now we have to compute the intersection between all pairs of
    // edges from the face 1 and from the face2.
    for (var j = 0u; j < face2.num_vertices; j++) {
        for (var i = 0u; i < face1.num_vertices; i++) {
            let bcoords = closest_points_line2d(
                projected_face1[i],
                projected_face1[(i + 1u) % face1.num_vertices],
                projected_face2[j],
                projected_face2[(j + 1u) % face2.num_vertices],
            );
            if bcoords.x > 0.0 && bcoords.x < 1.0 && bcoords.y > 0.0 && bcoords.y < 1.0 {
                // Found a contact between the two edges.
                let edge1_a = face1.vertices[i];
                let edge1_b = face1.vertices[(i + 1u) % face1.num_vertices];
                let edge2_a = vertices2_1[j];
                let edge2_b = vertices2_1[(j + 1u) % face2.num_vertices];
                let local_p1 = edge1_a * (1.0 - bcoords.x) + edge1_b * bcoords.x;
                let local_p2_1 = edge2_a * (1.0 - bcoords.y) + edge2_b * bcoords.y;
                let dist = dot(local_p2_1 - local_p1, sep_axis1);

                if dist <= prediction {
                    if !flipped {
                        candidates[num_candidates].pt = local_p1;
                        candidates[num_candidates].dist = dist;
                    } else {
                        let local_p2 = Pose::invMulPt(pose12, local_p2_1);
                        candidates[num_candidates].pt = local_p2;
                        candidates[num_candidates].dist = dist;
                    }
                    num_candidates += 1u;
                }

                if num_candidates == MAX_CANDIDATE_POINTS {
                    return manifold_reduction(candidates, num_candidates, sep_axis1);
                }
            }
        }
    }

    return manifold_reduction(candidates, num_candidates, sep_axis1);
}

fn manifold_reduction(candidates: array<Manifold::ContactPoint, MAX_CANDIDATE_POINTS>, num_candidates: u32, normal: Vector) -> Manifold::ContactManifold {
    var result = Manifold::ContactManifold();
    if num_candidates <= Manifold::MAX_MANIFOLD_POINTS {
        result.points_a[0] = candidates[0];
        result.points_a[1] = candidates[1];
        result.points_a[2] = candidates[2];
        result.points_a[3] = candidates[3];
        result.len = num_candidates;
        return result;
    }

    // Run contact reduction so we only have up to four solver contacts.
    // We follow a logic similar to Bepu’s approach.
    // 1. Find the deepest contact.
    var deepest_dist = candidates[0].dist;
    var selected = array(0, MAX_CANDIDATE_POINTS, MAX_CANDIDATE_POINTS, MAX_CANDIDATE_POINTS);

    for (var i = 1u; i < num_candidates; i++) {
        if candidates[i].dist < deepest_dist {
            deepest_dist = candidates[i].dist;
            selected[0] = i;
        }
    }

    // 2. Find the point that is the furthest from the deepest one.
    let selected_a = candidates[selected[0]].pt;
    var furthest_dist = -1.0e10;

    for (var i = 0u; i < num_candidates; i++) {
        let pt_sel = selected_a - candidates[i].pt;
        let dist = dot(pt_sel, pt_sel);
        if i != selected[0] && dist > furthest_dist
        {
            furthest_dist = dist;
            selected[1] = i;
        }
    }

    // 3. Now find the two points furthest from the segment we built so far.
    let selected_b = candidates[selected[1]].pt;
    let selected_ab = selected_b - selected_a;
    let tangent = cross(selected_ab, normal);

    // Find the points that minimize and maximize the dot product with the tangent.
    var min_dot = 1.0e10;
    var max_dot = -1.0e10;

    for (var i = 0u; i < num_candidates; i++) {
        if i == selected[0] || i == selected[1] {
            continue;
        }

        let dot = dot(candidates[i].pt - selected_a, tangent);
        if dot < min_dot {
            min_dot = dot;
            selected[2] = i;
        }

        if dot > max_dot {
            max_dot = dot;
            selected[3] = i;
        }
    }

    if selected[2] == MAX_CANDIDATE_POINTS {
        selected[2] = selected[3];
        selected[3] = MAX_CANDIDATE_POINTS;
    }

    result.points_a[0] = candidates[selected[0]];
    result.points_a[1] = candidates[selected[1]];
    result.len = 2u;

    if selected[2] != MAX_CANDIDATE_POINTS {
        result.points_a[2] = candidates[selected[2]];
        result.len = 3u;

        if selected[3] != MAX_CANDIDATE_POINTS {
            result.points_a[3] = candidates[selected[3]];
            result.len = 4u;
        }
    }

    return result;
}

/// Compute the barycentric coordinates of the intersection between the two given lines.
/// Returns `vec2(MAX_FLT, MAX_FLT)` if the lines are parallel.
fn closest_points_line2d(
    edge1_a: vec2<f32>,
    edge1_b: vec2<f32>,
    edge2_a: vec2<f32>,
    edge2_b: vec2<f32>,
) -> vec2<f32> {
    // Inspired by Real-time collision detection by Christer Ericson.
    let dir1 = edge1_b - edge1_a;
    let dir2 = edge2_b - edge2_a;
    let r = edge1_a - edge2_a;

    let a = dot(dir1, dir1);
    let e = dot(dir2, dir2);
    let f = dot(dir2, r);

    if a <= EPSILON && e <= EPSILON {
        return vec2(0.0, 0.0);
    } else if a <= EPSILON {
        return vec2(0.0, f / e);
    } else {
        let c = dot(dir1, r);
        if e <= EPSILON {
            return vec2(-c / a, 0.0);
        } else {
            let b = dot(dir1, dir2);
            let ae = a * e;
            let bb = b * b;
            let denom = ae - bb;

            // Use absolute and ulps error to test collinearity.
            // TODO: implement ulps equality check
            let parallel = denom <= EPSILON; // || approx::ulps_eq!(ae, bb);

            if !parallel {
                let s = (b * f - c * e) / denom;
                let t = (b * s + f) / e;
                return vec2(s, t);
            } else {
                return vec2(MAX_FLT, MAX_FLT);
            }
        }
    }
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

/// Returns the barycentric coordinates of the closest point on each segment.
///
/// If `bcoords` are the returned barycentric coordinates, the closest points
/// can be calculated as: `seg1_a * (1.0 - bcoords.x) + seg1_b * bcoords.x` on the first segment
/// and `seg2_a * (1.0 - bcoords.y) + seg2_b * bcoords.y` on the second segment.
fn closest_points_segment_segment(
    seg1_a: vec2<f32>,
    seg1_b: vec2<f32>,
    seg2_a: vec2<f32>,
    seg2_b: vec2<f32>,
) -> vec2<f32> {
    // Inspired by real-time collision detection by Christer Ericson.
    let d1 = seg1_b - seg1_a;
    let d2 = seg2_b - seg2_a;
    let r = seg1_a - seg2_a;

    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    var s = 0.0;
    var t = 0.0;

    // TODO PERF: make all these less branchy
    if a <= EPSILON && e <= EPSILON {
        s = 0.0;
        t = 0.0;
    } else if a <= EPSILON {
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if e <= EPSILON {
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let ae = a * e;
            let bb = b * b;
            let denom = ae - bb;

            // Use absolute and ulps error to test collinearity.
            // TODO: implement ULPS comparison.
            if denom > EPSILON { //&& !ulps_eq!(ae, bb) {
                s = clamp((b * f - c * e) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }

            t = (b * s + f) / e;

            if t < 0.0 {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }

    return vec2(s, t);
}

struct ClippingPoints {
    seg1_a: vec3<f32>,
    seg2_a: vec3<f32>,
    seg1_b: vec3<f32>,
    seg2_b: vec3<f32>,
    empty: bool,
}

/// Projects two segments on one another and compute their intersection.
fn clip_segment_segment(
    seg1_a_: vec3<f32>,
    seg1_b_: vec3<f32>,
    seg2_a_: vec3<f32>,
    seg2_b_: vec3<f32>,
) -> ClippingPoints {
    // NOTE: no need to normalize the tangent.
    var seg1_a = seg1_a_; // make the inputs mutable (for swapping)
    var seg1_b = seg1_b_; // make the inputs mutable (for swapping)
    var seg2_a = seg2_a_; // make the inputs mutable (for swapping)
    var seg2_b = seg2_b_; // make the inputs mutable (for swapping)

    var result = ClippingPoints();
    let tangent1 = seg1_b - seg1_a;
    let sqnorm_tangent1 = dot(tangent1, tangent1);

    var range1 = array(0.0, sqnorm_tangent1);
    var range2 = array(
        dot(seg2_a - seg1_a, tangent1),
        dot(seg2_b - seg1_a, tangent1),
    );

    if range1[1] < range1[0] {
        // Swap
        let swp1_a = seg1_a;
        seg1_a = seg1_b;
        seg1_b = swp1_a;
        range1 = array(range1[1], range1[0]);
    }

    if range2[1] < range2[0] {
        // Swap
        let swp2_a = seg2_a;
        seg2_a = seg2_b;
        seg2_b = swp2_a;
        range2 = array(range2[1], range2[0]);
    }

    if range2[0] > range1[1] || range1[0] > range2[1] {
        // No clip point.
        result.empty = true;
        return result;
    }

    let length1 = range1[1] - range1[0];
    let length2 = range2[1] - range2[0];

    if range2[0] > range1[0] {
        let bcoord = (range2[0] - range1[0]) / length1;
        result.seg1_a = seg1_a + tangent1 * bcoord;
        result.seg2_a = seg2_a;
    } else {
        let bcoord = (range1[0] - range2[0]) / length2;
        result.seg1_a = seg1_a;
        result.seg2_a = seg2_a + (seg2_b - seg2_a) * bcoord;
    }

    if range2[1] < range1[1] {
        let bcoord = (range2[1] - range1[0]) / length1;
        result.seg1_b = seg1_a + tangent1 * bcoord;
        result.seg2_b = seg2_b;
    } else {
        let bcoord = (range1[1] - range2[0]) / length2;
        result.seg1_b = seg1_b;
        result.seg2_b = seg2_a + (seg2_b - seg2_a) * bcoord;
    }

    result.empty = false;
    return result;
}

// TODO: share this from another module.
fn perp(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

// NOTE: ported from the `approx` rust crate.
fn relative_eq(a: f32, b: f32) -> bool {
    let abs_diff = abs(a - b);

    // For when the numbers are really close together
    if abs_diff <= EPSILON {
        return true;
    }

    let abs_a = abs(a);
    let abs_b = abs(b);

    // Use a relative difference comparison
    return abs_diff <= max(abs_b, abs_a) * EPSILON;
}
#endif
