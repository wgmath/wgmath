//! Radix Sort Indirect Dispatch Initialization
//!
//! This single-threaded kernel computes the workgroup counts for the radix sort pipeline.
//! It calculates how many workgroups are needed for the main sort passes and for the
//! reduction passes based on the input size.
//!
//! Outputs:
//! - num_wgs: Workgroups for main sort kernels (count, scatter)
//! - num_reduce_wgs: Workgroups for reduction kernels (reduce, scan, scan_add)
//!
//! Called once at the beginning of each sort operation to set up indirect dispatches.

#import wgparry::utils::sorting as sorting;

@group(0) @binding(0) var<storage, read> len: u32;
@group(0) @binding(1) var<storage, read_write> num_wgs: array<u32>;
@group(0) @binding(2) var<storage, read_write> num_reduce_wgs: array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if invocation_id.x != 0 {
        return;
    }

    let wgs = sorting::div_ceil(len, sorting::BLOCK_SIZE);
    num_wgs[0] = wgs;
    num_wgs[1] = 1;
    num_wgs[2] = 1;
    num_reduce_wgs[0] = sorting::div_ceil(wgs, sorting::BLOCK_SIZE) * sorting::BIN_COUNT;
    num_reduce_wgs[1] = 1;
    num_reduce_wgs[2] = 1;
}