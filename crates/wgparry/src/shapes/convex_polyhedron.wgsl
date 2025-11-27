#define_import_path wgparry::convex

#import wgparry::polygonal_feature as Feat
#import wgparry::bounding_volumes::aabb as Aabb
#import wgparry::vtx_idx as VtxIdx

struct ConvexPolyhedron {
    // For finding support points.
    first_vtx_id: u32,
    end_vtx_id: u32,
    // For finding support faces (3D only).
    first_face_id: u32,
    end_face_id: u32,
}

// TODO: cache the AABB (for example as the first two entries of the shape’s index buffer)
//       so it doesn’t get recomputed at each frame?
fn aabb(shape: ConvexPolyhedron) -> Aabb::Aabb {
    var mins = VtxIdx::vertices[shape.first_vtx_id];
    var maxs = VtxIdx::vertices[shape.first_vtx_id];

    for (var i = shape.first_vtx_id; i != shape.end_vtx_id; i++) {
        mins = min(mins, VtxIdx::vertices[i]);
        maxs = max(maxs, VtxIdx::vertices[i]);
    }

    return Aabb::Aabb(mins, maxs);
}

fn local_support_point(shape: ConvexPolyhedron, dir: Vector) -> Vector {
    var best_dot = -1.0e20;
    var best = Vector();
    for (var i = shape.first_vtx_id; i != shape.end_vtx_id; i++) {
        let val = dot(VtxIdx::vertices[i], dir);
        if val > best_dot {
            best_dot = val;
            best = VtxIdx::vertices[i];
        }
    }

    return best;
}

#if DIM == 2
fn support_face(shape: ConvexPolyhedron, dir: Vector) -> Feat::PolygonalFeature {
    var result = Feat::PolygonalFeature();
    var best = vec2(0u);
    var best_dot = -1.0e20;
    let num_vertices = shape.end_vtx_id - shape.first_vtx_id;

    for (var i = 0u; i < num_vertices; i += 1u) {
        let j = (i + 1u) % num_vertices;
        let a = VtxIdx::vertices[shape.first_vtx_id + i];
        let b = VtxIdx::vertices[shape.first_vtx_id + j];
        let ab = b - a;
        // CounterClockWise 2D normal.
        let n = vec2(ab.y, -ab.x);
        let n_len = length(n);

        if n_len != 0.0 {
            let val = dot(n / n_len, dir);
            if val > best_dot {
                best_dot = val;
                best = vec2(i, j);
            }
        }
    }

    result.vertices[0] = VtxIdx::vertices[shape.first_vtx_id + best.x];
    result.vertices[1] = VtxIdx::vertices[shape.first_vtx_id + best.y];
    result.num_vertices = 2;
    return result;
}
#else
fn support_face(shape: ConvexPolyhedron, dir: Vector) -> Feat::PolygonalFeature {
    var result = Feat::PolygonalFeature();
    var best = vec3(0u);
    var best_dot = -1.0e20;
    let base_vid = vec3(shape.first_vtx_id);

    for (var i = shape.first_face_id; i != shape.end_face_id; i += 3) {
        let vids = base_vid + vec3(VtxIdx::indices[i], VtxIdx::indices[i + 1], VtxIdx::indices[i + 2]);
        let a = VtxIdx::vertices[vids.x];
        let b = VtxIdx::vertices[vids.y];
        let c = VtxIdx::vertices[vids.z];
        let ab = b - a;
        let ac = c - a;
        let n = cross(ab, ac);
        let n_len = length(n);

        if n_len != 0.0 {
            let val = dot(n / n_len, dir);
            if val > best_dot {
                best_dot = val;
                best = vids;
            }
        }
    }

    result.vertices[0] = VtxIdx::vertices[best.x];
    result.vertices[1] = VtxIdx::vertices[best.y];
    result.vertices[2] = VtxIdx::vertices[best.z];
    result.num_vertices = 3;
    return result;
}
#endif