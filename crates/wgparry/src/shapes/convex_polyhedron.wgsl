#define_import_path wgparry::convex

#import wgparry::polygonal_feature as Feat
#import wgparry::bounding_volumes::aabb as Aabb

@group(2) @binding(0)
var<storage, read> vertices: array<Vector>;
@group(2) @binding(1)
var<storage, read> indices: array<u32>;

struct ConvexPolyhedron {
    // For finding support points.
    first_vtx_id: u32,
    end_vtx_id: u32,
    // For finding support faces.
    first_face_id: u32,
    end_face_id: u32,
}

// TODO: cache the AABB (for example as the first two entries of the shape’s index buffer)
//       so it doesn’t get recomputed at each frame?
fn aabb(shape: ConvexPolyhedron) -> Aabb::Aabb {
    var mins = vertices[shape.first_vtx_id];
    var maxs = vertices[shape.first_vtx_id];

    for (var i = shape.first_vtx_id; i != shape.end_vtx_id; i++) {
        mins = min(mins, vertices[i]);
        maxs = max(maxs, vertices[i]);
    }

    return Aabb::Aabb(mins, maxs);
}

fn local_support_point(shape: ConvexPolyhedron, dir: Vector) -> Vector {
    var best_dot = -1.0e20;
    var best = Vector();
    for (var i = shape.first_vtx_id; i != shape.end_vtx_id; i++) {
        let val = dot(vertices[i], dir);
        if val > best_dot {
            best_dot = val;
            best = vertices[i];
        }
    }

    return best;
}

fn support_face(shape: ConvexPolyhedron, dir: Vector) -> Feat::PolygonalFeature {
    var result = Feat::PolygonalFeature();
    var best = vec3(0u);
    var best_dot = -1.0e20;
    let base_vid = vec3(shape.first_vtx_id);

    for (var i = shape.first_face_id; i != shape.end_face_id; i += 3) {
        let vids = base_vid + vec3(indices[i], indices[i + 1], indices[i + 2]);
        let a = vertices[vids.x];
        let b = vertices[vids.y];
        let c = vertices[vids.z];
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

    result.vertices[0] = vertices[best.x];
    result.vertices[1] = vertices[best.y];
    result.vertices[2] = vertices[best.z];
    result.num_vertices = 3;
    return result;
}