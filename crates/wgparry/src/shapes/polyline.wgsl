#define_import_path wgparry::polyline

#import wgparry::polygonal_feature as Feat
#import wgparry::bounding_volumes::aabb as Aabb
#import wgparry::segment as Segment
#import wgparry::vtx_idx as VtxIdx

/// A polyline
///
/// The polyline’s vertex and index buffer are organized is a way that the vertex
/// buffer contains the BVH first, and then then the segment vertices.
/// Similarly, its index buffer contains the BVH topology information fisrt, and then
/// the segment vertices.
///
/// The BVH topology follows https://docs.rs/bvh/0.12.0/bvh/flat_bvh/struct.FlatNode.html
/// So each BVH node implies 3 entries in the index buffer, and 2 entries in the vertex buffer.
struct Polyline {
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
fn aabb(shape: Polyline) -> Aabb::Aabb {
    return shape.root_aabb;
}

fn bvh_node_aabb(mesh: Polyline, node_id: u32) -> Aabb::Aabb {
    let vid = mesh.bvh_vtx_root_id + node_id * 2u; // Multiply by 2 since there are two values per AABB (min/max).
    return Aabb::Aabb(VtxIdx::vertices[vid], VtxIdx::vertices[vid + 1]);
}

fn bvh_node_idx(mesh: Polyline, node_id: u32) -> BvhIdx {
    let base_id = mesh.bvh_idx_root_id + node_id * 3u;
    return BvhIdx(VtxIdx::indices[base_id], VtxIdx::indices[base_id + 1], VtxIdx::indices[base_id + 2]);
}

fn segment(mesh: Polyline, seg_id: u32) -> Segment::Segment {
    let base_id = mesh.bvh_idx_root_id + mesh.bvh_node_len * 3u + seg_id * 2u;
    let base_vid = mesh.bvh_vtx_root_id + mesh.bvh_node_len * 2u;
    let a = VtxIdx::vertices[base_vid + VtxIdx::indices[base_id]];
    let b = VtxIdx::vertices[base_vid + VtxIdx::indices[base_id + 1]];
    return Segment::Segment(a, b);
}

// TODO PERF: tree traversal is implemented directly in `narrow_phase.wgsl` so it can redispatch
//       the collision algorithm once a leaf is reached.
//       An alternative would be to collect leaves in an `array<u32, MAX_HITS>` but then
//       we’d be limited to `MAX_HITS` or would need more complex logic. So let’s start
//       for the simpler approach until we get to benchmark alternatives.