#import wgblas::shape as Shape;

@group(0) @binding(0)
var<uniform> shape: Shape::Shape;
@group(0) @binding(1)
var<storage, read> input: array<f32>;
@group(0) @binding(2)
var<storage, read_write> output: f32;

const WORKGROUP_SIZE: u32 = 128;

fn reduce_sum_f32(acc: f32, x: f32) -> f32 {
    return acc + x;
}

fn reduce_prod_f32(acc: f32, x: f32) -> f32 {
    return acc * x;
}

fn reduce_min_f32(acc: f32, x: f32) -> f32 {
    return min(acc, x);
}

fn reduce_max_f32(acc: f32, x: f32) -> f32 {
    return max(acc, x);
}

fn reduce_sqnorm_f32(acc: f32, x: f32) -> f32 {
    return acc + x * x;
}

fn init_zero() -> f32 {
    return 0.0;
}

fn init_one() -> f32 {
    return 1.0;
}

fn init_max_f32() -> f32 {
    return 3.4e38; // TODO: the exact value isn’t accepted on the web: 3.40282347E+38;
}

fn init_min_f32() -> f32 {
    return -3.4e38; // TODO: the exact value isn’t accepted on the web: 3.40282347E+38;
}

fn init_placeholder() -> f32 {
    return 0.0;
}

fn reduce_placeholder(acc: f32, x: f32) -> f32 {
    return acc + x;
}
fn workspace_placeholder(acc: f32, x: f32) -> f32 {
    return acc + x;
}

var<workgroup> workspace: array<f32, WORKGROUP_SIZE>;

fn reduce(thread_id: u32, stride: u32) {
    if thread_id < stride {
        workspace[thread_id] = reduce_placeholder(workspace[thread_id], workspace[thread_id + stride]);
    }
    workgroupBarrier();
}

fn run_reduction(thread_id: u32) -> f32 {
    workspace[thread_id] = init_placeholder();

    for (var i = thread_id; i < shape.nrows; i += WORKGROUP_SIZE) {
        let val_i = input[Shape::iv(shape, i)];
        workspace[thread_id] = workspace_placeholder(workspace[thread_id], val_i);
    }

    workgroupBarrier();

    reduce(thread_id, 64u);
    reduce(thread_id, 32u);
    reduce(thread_id, 16u);
    reduce(thread_id, 8u);
    reduce(thread_id, 4u);
    reduce(thread_id, 2u);
    reduce(thread_id, 1u);

    return workspace[0];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let result = run_reduction(invocation_id.x);

    if (invocation_id.x == 0) {
        output = result;
    }
}
