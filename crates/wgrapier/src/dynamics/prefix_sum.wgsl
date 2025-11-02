//! Parallel prefix sum  implementation
//!
//! This shader implements an efficient parallel prefix sum (also called cumulative sum).
//!
//! What is Prefix Sum?
//! Given an input array [a0, a1, a2, ...], compute output [0, a0, a0+a1, a0+a1+a2, ...]
//! Each output element is the sum of the element with all elements before it.
//!
//! Use in `wgrapier`:
//! Used to compute per-body constraint ranges from constraint counts:
//! - Input: [3, 2, 5, 1] (number of constraints per body)
//! - Output: [0, 3, 5, 10] (end index of constraints for each body)
//! - Body i's constraints are at indices [output[i-1], output[i])

#define_import_path wgsparkl::grid::prefix_sum

/// Input/output data array for prefix sum.
/// On input: values to sum. On output: exclusive prefix sum.
@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

/// Auxiliary array for storing per-block totals.
/// Used to propagate sums across multiple workgroups.
@group(0) @binding(1)
var<storage, read_write> aux: array<u32>;

/// Workgroup size: number of elements processed per workgroup.
const WORKGROUP_SIZE: u32 = 256;

/// Shared memory workspace for parallel scan within a workgroup.
var<workgroup> workspace: array<u32, WORKGROUP_SIZE>;

/// Performs exclusive prefix sum on a segment of the data array.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prefix_sum(@builtin(local_invocation_id) thread_id: vec3<u32>, @builtin(workgroup_id) block_id: vec3<u32>) {
    let bid = block_id.x;
    let tid = thread_id.x;
    let data_len = arrayLength(&data);

    // Early exit if this workgroup is beyond the data range
    if bid * WORKGROUP_SIZE >= data_len {
        return;
    }

    // Compute the number of elements this block will process
    let data_block_len = data_len - bid * WORKGROUP_SIZE;

    // Round up to next power of two for algorithm efficiency (required for tree structure)
    let shared_len = clamp(next_power_of_two(data_block_len), 1u, WORKGROUP_SIZE);

    // Global index for this thread's element
    let elt_id = tid + bid * WORKGROUP_SIZE;

    // Phase 0: Load data into shared memory
    if elt_id < data_len {
        workspace[tid] = data[elt_id];
    } else {
        // Pad with zeros for out-of-bounds threads
        workspace[tid] = 0u;
    }

    // Phase 1: Up-sweep (reduce) - build tree of partial sums
    // After this phase, workspace[shared_len-1] contains the total sum
    {
        var d = shared_len / 2;  // Number of active threads at this level
        var offset = 1u;          // Stride between elements at this level
        while (d > 0) {
            workgroupBarrier();   // Synchronize before reading shared memory
            if tid < d {
                // Compute indices for the two elements to sum
                let ia = tid * 2u * offset + offset - 1u;
                let ib = (tid * 2u + 1u) * offset + offset - 1u;

                // Sum and store in right child (ib)
                let sum = workspace[ia] + workspace[ib];
                workspace[ib] = sum;
            }

            // Move up one level in the tree
            d /= 2u;
            offset *= 2u;
        }
    }

    // Thread 0 saves the total sum and clears the root for down-sweep
    if tid == 0 {
        let total_sum = workspace[shared_len - 1];
        aux[bid] = total_sum;  // Save for multi-block handling
        workspace[shared_len - 1] = 0u;  // Initialize root for down-sweep
    }

    // Phase 2: Down-sweep - propagate partial sums down the tree
    // Transforms the tree into an exclusive scan
    {
        var d = 1u;                    // Number of active threads at this level
        var offset = shared_len / 2u;  // Stride between elements at this level
        while (d < shared_len) {
            workgroupBarrier();   // Synchronize before reading shared memory
            if tid < d {
                // Compute indices for the two elements to update
                let ia = tid * 2u * offset + offset - 1u;
                let ib = (tid * 2u + 1u) * offset + offset - 1u;

                let a = workspace[ia];
                let b = workspace[ib];

                // Swap and accumulate: left gets parent's value, right gets parent + left
                workspace[ia] = b;
                workspace[ib] = a + b;
            }

            // Move down one level in the tree
            d *= 2u;
            offset /= 2u;
        }
    }

    // Synchronize before writing results
    workgroupBarrier();

    // Write results back to global memory
    if elt_id < data_len {
        data[elt_id] = workspace[tid];
    }
}

/// Adds per-block offsets to complete multi-block prefix sum.
///
/// After each block computes its local prefix sum, we need to add the total
/// sum from all previous blocks to each element. This kernel adds aux[bid-1]
/// (the sum of all blocks before this one) to each element in block bid.
///
/// @param thread_id: Global thread ID
/// @param block_id: Workgroup ID
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn add_data_grp(@builtin(global_invocation_id) thread_id: vec3<u32>, @builtin(workgroup_id) block_id: vec3<u32>) {
    let tid = thread_id.x;
    let bid = block_id.x;
    if tid < arrayLength(&data) {
        // Add the cumulative sum from all previous blocks
        data[tid] += aux[bid];
    }
}

/// Computes the next power of two greater than or equal to val.
///
/// Uses bit manipulation tricks to efficiently round up to power of two.
/// Required for the tree-based scan algorithm which needs power-of-two sizes.
///
/// See Bit Twiddling Hacks: https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
///
/// @param val: Input value
/// @returns: Smallest power of two >= val
fn next_power_of_two(val: u32) -> u32 {
    var v = val;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}