#define_import_path wgsparkl::grid::prefix_sum

@group(0) @binding(0)
var<storage, read_write> data: array<u32>;
@group(0) @binding(1)
var<storage, read_write> aux: array<u32>;

const WORKGROUP_SIZE: u32 = 256;
var<workgroup> workspace: array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prefix_sum(@builtin(local_invocation_id) thread_id: vec3<u32>, @builtin(workgroup_id) block_id: vec3<u32>) {
    let bid = block_id.x;
    let tid = thread_id.x;
    let data_len = arrayLength(&data);

    if bid * WORKGROUP_SIZE >= data_len {
        return;
    }

    let data_block_len = data_len - bid * WORKGROUP_SIZE;
    let shared_len = clamp(next_power_of_two(data_block_len), 1u, WORKGROUP_SIZE);
    let elt_id = tid + bid * WORKGROUP_SIZE;

    // Init the shared memory.
    if elt_id < data_len {
        workspace[tid] = data[elt_id];
    } else {
        workspace[tid] = 0u;
    }

    // Up-sweep.
    {
        var d = shared_len / 2;
        var offset = 1u;
        while (d > 0) {
            workgroupBarrier();
            if tid < d {
                let ia = tid * 2u * offset + offset - 1u;
                let ib = (tid * 2u + 1u) * offset + offset - 1u;

                let sum = workspace[ia] + workspace[ib];
                workspace[ib] = sum;
            }

            d /= 2u;
            offset *= 2u;
        }
    }

    if tid == 0 {
        let total_sum = workspace[shared_len - 1];
        aux[bid] = total_sum;
        workspace[shared_len - 1] = 0u;
    }

    // Down-sweep.
    {
        var d = 1u;
        var offset = shared_len / 2u;
        while (d < shared_len) {
            workgroupBarrier();
            if tid < d {
                let ia = tid * 2u * offset + offset - 1u;
                let ib = (tid * 2u + 1u) * offset + offset - 1u;

                let a = workspace[ia];
                let b = workspace[ib];

                workspace[ia] = b;
                workspace[ib] = a + b;
            }

            d *= 2u;
            offset /= 2u;
        }
    }

    workgroupBarrier();

    if elt_id < data_len {
        data[elt_id] = workspace[tid];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn add_data_grp(@builtin(global_invocation_id) thread_id: vec3<u32>, @builtin(workgroup_id) block_id: vec3<u32>) {
    let tid = thread_id.x;
    let bid = block_id.x;
    if tid < arrayLength(&data) {
        data[tid] += aux[bid];
    }
}

// See Bit Twiddling Hack: https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
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