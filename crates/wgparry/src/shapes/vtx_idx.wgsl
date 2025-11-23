//! Buffer bindings for all the complex shapes requiring an index and vertex buffer
//! (trimesh, convex polyhedrons, etc.)

#define_import_path wgparry::vtx_idx

@group(2) @binding(0)
var<storage, read> vertices: array<Vector>;
@group(2) @binding(1)
var<storage, read> indices: array<u32>;