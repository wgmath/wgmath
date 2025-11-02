//! Radix Sort Count (Histogram) Kernel
//!
//! First pass of radix sort: computes per-workgroup histograms for the current 4-bit digit.
//!
//! Algorithm:
//! 1. Each workgroup processes BLOCK_SIZE (1024) consecutive elements
//! 2. Initialize shared memory histogram to zeros
//! 3. Each thread processes ELEMENTS_PER_THREAD (4) elements
//! 4. Extract 4-bit key from current shift position
//! 5. Atomically increment corresponding histogram bin
//! 6. Write per-workgroup histogram to global memory
//!
//! Output layout: counts[bin * num_workgroups + workgroup_id]
//! This produces num_workgroups separate histograms, one per workgroup.
//!
//! Workgroup size: 256 threads
//! Shared memory: 16 atomic counters (one per bin)

#import wgparry::utils::sorting as sorting;

struct Uniforms {
    /// Bit shift amount for this sort pass (0, 4, 8, 12, ..., 28).
    shift: u32,
}

@group(0) @binding(0) var<storage, read> config: Uniforms;
@group(0) @binding(1) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;

/// Shared memory histogram for atomic accumulation.
var<workgroup> histogram: array<atomic<u32>, sorting::BIN_COUNT>;

@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>,
) {
    let num_keys = num_keys_arr[0];

    // let num_keys = num_keys_arr[0];
    let num_wgs = sorting::div_ceil(num_keys, sorting::BLOCK_SIZE);
    let group_id = gid.x;

    if group_id >= num_wgs {
        return;
    }

    if local_id.x < sorting::BIN_COUNT {
        histogram[local_id.x] = 0u;
    }
    workgroupBarrier();

    let wg_block_start = sorting::BLOCK_SIZE * group_id;
    var block_index = wg_block_start + local_id.x;
    let shift_bit = config.shift;
    var data_index = block_index;

    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        if data_index < num_keys {
            let local_key = (src[data_index] >> shift_bit) & 0xfu;
            atomicAdd(&histogram[local_key], 1u);
        }
        data_index += sorting::WG;
    }
    block_index += sorting::BLOCK_SIZE;
    workgroupBarrier();
    if local_id.x < sorting::BIN_COUNT {
        let num_wgs = sorting::div_ceil(num_keys, sorting::BLOCK_SIZE);
        counts[local_id.x * num_wgs + group_id] = histogram[local_id.x];
    }
}
