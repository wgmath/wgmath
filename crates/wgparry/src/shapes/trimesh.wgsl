#define_import_path wgparry::trimesh

#import wgparry::polygonal_feature as Feat
#import wgparry::bounding_volumes::aabb as Aabb
#import wgparry::triangle as Triangle
#import wgparry::vtx_idx as VtxIdx

/// A triangle mesh.
///
/// The mesh’s vertex and index buffer are organized is a way that the vertex
/// buffer contains the BVH first, and then then the triangle vertices.
/// Similarly, its index buffer contains the BVH topology information fisrt, and then
/// the triangle vertices.
///
/// The mesh topology follows https://docs.rs/bvh/0.12.0/bvh/flat_bvh/struct.FlatNode.html
/// So each BVH node implies 3 entries in the index buffer, and 2 entries in the vertex buffer.
struct TriMesh {
    /// Index of the root AABB in the vertex buffer.
    bvh_vtx_root_id: u32,
    /// The root AABB left-child index.
    bvh_idx_root_id: u32,
    // The number of BVH nodes. Triangle indices are stored after the last bvh node.
    bvh_node_len: u32,
    root_aabb: Aabb::Aabb,
}

struct BvhIdx {
    entry_index: u32,
    exit_index: u32,
    shape_index: u32,
}

// Simply return the root aabb.
fn aabb(shape: TriMesh) -> Aabb::Aabb {
    return shape.root_aabb;
}

fn bvh_node_aabb(mesh: TriMesh, node_id: u32) -> Aabb::Aabb {
    let vid = mesh.bvh_vtx_root_id + node_id * 2u; // Multiply by 2 since there are two values per AABB (min/max).
    return Aabb::Aabb(VtxIdx::vertices[vid], VtxIdx::vertices[vid + 1]);
}

fn bvh_node_idx(mesh: TriMesh, node_id: u32) -> BvhIdx {
    let base_id = mesh.bvh_idx_root_id + node_id * 3u;
    return BvhIdx(VtxIdx::indices[base_id], VtxIdx::indices[base_id + 1], VtxIdx::indices[base_id + 2]);
}

fn triangle(mesh: TriMesh, tri_id: u32) -> Triangle::Triangle {
    let base_id = mesh.bvh_idx_root_id + mesh.bvh_node_len * 3u + tri_id * 3u;
    let base_vid = mesh.bvh_vtx_root_id + mesh.bvh_node_len * 2u;
    let a = VtxIdx::vertices[base_vid + VtxIdx::indices[base_id]];
    let b = VtxIdx::vertices[base_vid + VtxIdx::indices[base_id + 1]];
    let c = VtxIdx::vertices[base_vid + VtxIdx::indices[base_id + 2]];
    return Triangle::Triangle(a, b, c);
}

// TODO PERF: tree traversal is implemented directly in `narrow_phase.wgsl` so it can redispatch
//       the collision algorithm once a leaf is reached.
//       An alternative would be to collect leaves in an `array<u32, MAX_HITS>` but then
//       we’d be limited to `MAX_HITS` or would need more complex logic. So let’s start
//       for the simpler approach until we get to benchmark alternatives.