//! Buffer bindings for all the complex shapes requiring an index and vertex buffer
//! (trimesh, convex polyhedrons, etc.)

use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;

#[derive(Shader)]
#[shader(
    src = "vtx_idx.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Buffer bindings for index-buffers and vertex-buffers.
pub struct WgVtxIdx;
