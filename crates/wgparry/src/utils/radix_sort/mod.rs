//! Radix sort implementation, ported from `brush-sort`: https://github.com/ArthurBrussee/brush/tree/main/crates/brush-sort

use naga_oil::compose::ComposerError;
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::Shader;
use wgpu::util::DispatchIndirectArgs;
use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

// NOTE: must match the values from `sorting.wgsl`.
const WG: u32 = 256;
const ELEMENTS_PER_THREAD: u32 = 4;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1 << BITS_PER_PASS;

#[derive(Shader)]
#[shader(
    derive(Sorting),
    src = "./init_indirect_dispatches.wgsl",
    composable = false
)]
struct InitIndirectDispatches {
    main: ComputePipeline,
}

#[derive(Shader)]
#[shader(derive(Sorting), src = "./sort_count.wgsl", composable = false)]
struct SortCount {
    main: ComputePipeline,
}

#[derive(Shader)]
#[shader(derive(Sorting), src = "./sort_reduce.wgsl", composable = false)]
struct SortReduce {
    main: ComputePipeline,
}

#[derive(Shader)]
#[shader(derive(Sorting), src = "./sort_scan.wgsl", composable = false)]
struct SortScan {
    main: ComputePipeline,
}

#[derive(Shader)]
#[shader(derive(Sorting), src = "./sort_scan_add.wgsl", composable = false)]
struct SortScanAdd {
    main: ComputePipeline,
}

#[derive(Shader)]
#[shader(derive(Sorting), src = "./sort_scatter.wgsl", composable = false)]
struct SortScatter {
    main: ComputePipeline,
}

#[derive(Shader)]
#[shader(src = "./sorting.wgsl")]
struct Sorting;

pub struct RadixSort {
    init: InitIndirectDispatches,
    count: SortCount,
    reduce: SortReduce,
    scan: SortScan,
    scan_add: SortScanAdd,
    scatter: SortScatter,
}

pub struct RadixSortWorkspace {
    pass_uniforms: Vec<GpuScalar<u32>>,
    reduced_buf: GpuVector<u32>, // Tensor of size BLOCK_SIZE
    count_buf: GpuVector<u32>,
    num_wgs: GpuScalar<[u32; 3]>,
    num_reduce_wgs: GpuScalar<[u32; 3]>,
    output_keys_pong: GpuVector<u32>, // dual-buffering for output keys.
    output_values_pong: GpuVector<u32>, // dual-buffering for output values.
}

impl RadixSortWorkspace {
    pub fn new(device: &Device) -> Self {
        let zeros = vec![0u32; BLOCK_SIZE as usize];
        Self {
            pass_uniforms: vec![],
            reduced_buf: GpuVector::init(device, &zeros, BufferUsages::STORAGE),
            count_buf: GpuVector::uninit(device, 0, BufferUsages::STORAGE),
            num_wgs: GpuScalar::init(
                device,
                [1; 3],
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            ),
            num_reduce_wgs: GpuScalar::init(
                device,
                [1; 3],
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            ),
            output_keys_pong: GpuVector::uninit(device, 0, BufferUsages::STORAGE),
            output_values_pong: GpuVector::uninit(device, 0, BufferUsages::STORAGE),
        }
    }
}

impl RadixSort {
    pub fn from_device(device: &Device) -> Result<Self, ComposerError> {
        Ok(Self {
            init: InitIndirectDispatches::from_device(device)?,
            count: SortCount::from_device(device)?,
            reduce: SortReduce::from_device(device)?,
            scan: SortScan::from_device(device)?,
            scan_add: SortScanAdd::from_device(device)?,
            scatter: SortScatter::from_device(device)?,
        })
    }

    pub fn dispatch(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        workspace: &mut RadixSortWorkspace,
        input_keys: &GpuVector<u32>,
        input_values: &GpuVector<u32>,
        n_sort: &GpuScalar<u32>,
        sorting_bits: u32,
        output_keys: &GpuVector<u32>,
        output_values: &GpuVector<u32>,
    ) {
        assert_eq!(
            input_keys.len(),
            input_values.len(),
            "Input keys and values must have the same number of elements"
        );
        assert!(sorting_bits <= 32, "Can only sort up to 32 bits");

        let max_n = input_keys.len() as u32;

        // compute buffer and dispatch sizes
        let max_needed_wgs = max_n.div_ceil(BLOCK_SIZE);
        if workspace.count_buf.len() < max_needed_wgs as u64 * 16 {
            println!(">>> Gen nuw buf");
            workspace.count_buf =
                GpuVector::uninit(device, max_needed_wgs * 16, BufferUsages::STORAGE);
        }

        KernelDispatch::new(device, pass, &self.init.main)
            .bind0([
                n_sort.buffer(),
                workspace.num_wgs.buffer(),
                workspace.num_reduce_wgs.buffer(),
            ])
            .dispatch(1);

        let mut cur_keys = input_keys;
        let mut cur_vals = input_values;

        if workspace.output_keys_pong.len() < input_keys.len() {
            // TODO: is this OK even in the case where we call the radix sort multiple times
            //       successively but with increasing input buffer sizes? Wondering if that could
            //       free the previous buffer and then crash the previous invocation.
            workspace.output_keys_pong =
                GpuVector::uninit(device, input_keys.len() as u32, BufferUsages::STORAGE);
            workspace.output_values_pong =
                GpuVector::uninit(device, input_values.len() as u32, BufferUsages::STORAGE);
            println!(">>> Gen nuw buf");
        }

        let num_passes = sorting_bits.div_ceil(4);
        let mut output_keys = output_keys;
        let mut output_values = output_values;
        let mut output_keys_pong = &workspace.output_keys_pong;
        let mut output_values_pong = &workspace.output_values_pong;

        if num_passes % 2 == 0 {
            // Make sure the last pass has the user provided `output_keys`
            // set as the output buffer so that the final results doesn’t end
            // up stored in the workspace’s pong buffers instead.
            std::mem::swap(&mut output_keys, &mut output_keys_pong);
            std::mem::swap(&mut output_values, &mut output_values_pong);
        }

        for pass_id in 0..num_passes {
            if pass_id as usize >= workspace.pass_uniforms.len() {
                workspace.pass_uniforms.push(GpuScalar::init(
                    device,
                    pass_id * 4,
                    BufferUsages::STORAGE | BufferUsages::UNIFORM,
                ));
                println!(">>> Gen nuw buf");
            }

            let uniforms_buffer = &workspace.pass_uniforms[pass_id as usize];

            KernelDispatch::new(device, pass, &self.count.main)
                .bind0([
                    uniforms_buffer.buffer(),
                    n_sort.buffer(),
                    cur_keys.buffer(),
                    workspace.count_buf.buffer(),
                ])
                .dispatch_indirect(workspace.num_wgs.buffer());

            KernelDispatch::new(device, pass, &self.reduce.main)
                .bind0([
                    n_sort.buffer(),
                    workspace.count_buf.buffer(),
                    workspace.reduced_buf.buffer(),
                ])
                .dispatch_indirect(workspace.num_reduce_wgs.buffer());

            KernelDispatch::new(device, pass, &self.scan.main)
                .bind0([n_sort.buffer(), workspace.reduced_buf.buffer()])
                .dispatch(1);

            KernelDispatch::new(device, pass, &self.scan_add.main)
                .bind0([
                    n_sort.buffer(),
                    workspace.reduced_buf.buffer(),
                    workspace.count_buf.buffer(),
                ])
                .dispatch_indirect(workspace.num_reduce_wgs.buffer());

            KernelDispatch::new(device, pass, &self.scatter.main)
                .bind0([
                    uniforms_buffer.buffer(),
                    n_sort.buffer(),
                    cur_keys.buffer(),
                    cur_vals.buffer(),
                    workspace.count_buf.buffer(),
                    output_keys.buffer(),
                    output_values.buffer(),
                ])
                .dispatch_indirect(workspace.num_wgs.buffer());

            if pass_id == 0 {
                cur_keys = output_keys;
                cur_vals = output_values;
                output_keys = output_keys_pong;
                output_values = output_values_pong;
            } else {
                std::mem::swap(&mut cur_keys, &mut output_keys);
                std::mem::swap(&mut cur_vals, &mut output_values);
            }
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use crate::utils::radix_sort::RadixSortWorkspace;
    use crate::utils::RadixSort;
    use na::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::CommandEncoderExt;
    use wgcore::tensor::{GpuScalar, GpuVector};
    use wgpu::BufferUsages;

    pub fn cpu_argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| &data[i]);
        indices
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn test_sorting() {
        let gpu = GpuInstance::new().await.unwrap();
        let sort = RadixSort::from_device(gpu.device()).unwrap();
        let mut workspace = RadixSortWorkspace::new(gpu.device());

        for i in 0u32..128 {
            let keys_inp = [
                5 + i * 4,
                i,
                6,
                123,
                74657,
                123,
                999,
                2u32.pow(24) + 123,
                6,
                7,
                8,
                0,
                i * 2,
                16 + i,
                128 * i,
            ];

            let values_inp: Vec<_> = keys_inp.iter().copied().map(|x| x * 2 + 5).collect();

            let input_usages = BufferUsages::STORAGE;
            let output_usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
            let staging_usage = BufferUsages::MAP_READ | BufferUsages::COPY_DST;

            let keys = GpuVector::init(gpu.device(), &keys_inp, input_usages);
            let values = GpuVector::init(gpu.device(), &values_inp, input_usages);
            let out_keys = GpuVector::init(gpu.device(), &keys_inp, output_usages);
            let out_values = GpuVector::init(gpu.device(), &values_inp, output_usages);
            let staging_keys = GpuVector::init(gpu.device(), &keys_inp, staging_usage);
            let staging_values = GpuVector::init(gpu.device(), &values_inp, staging_usage);
            let num_points =
                GpuScalar::init(gpu.device(), keys_inp.len() as u32, BufferUsages::STORAGE);

            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            let mut pass = encoder.compute_pass("test", None);
            sort.dispatch(
                gpu.device(),
                &mut pass,
                &mut workspace,
                &keys,
                &values,
                &num_points,
                32,
                &out_keys,
                &out_values,
            );
            drop(pass);
            staging_keys.copy_from(&mut encoder, &out_keys);
            staging_values.copy_from(&mut encoder, &out_values);
            gpu.queue().submit(Some(encoder.finish()));

            let result_keys = staging_keys.read(gpu.device()).await.unwrap();
            let result_values = staging_values.read(gpu.device()).await.unwrap();

            let inds = cpu_argsort(&keys_inp);
            let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
            let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

            assert_eq!(DVector::from(ref_keys), DVector::from(result_keys));
            assert_eq!(DVector::from(ref_values), DVector::from(result_values));
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn test_sorting_big() {
        use rand::Rng;

        let gpu = GpuInstance::new().await.unwrap();
        let sort = RadixSort::from_device(gpu.device()).unwrap();
        let mut workspace = RadixSortWorkspace::new(gpu.device());

        // Simulate some data as one might find for a bunch of gaussians.
        let mut rng = rand::rng();
        let mut keys_inp = Vec::new();
        for i in 0..10000 {
            let start = rng.random_range(i..i + 150);
            let end = rng.random_range(start..start + 250);

            for j in start..end {
                if rng.random::<f32>() < 0.5 {
                    keys_inp.push(j);
                }
            }
        }
        let values_inp: Vec<_> = keys_inp.iter().map(|&x| x * 2 + 5).collect();

        let input_usages = BufferUsages::STORAGE;
        let output_usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let staging_usage = BufferUsages::MAP_READ | BufferUsages::COPY_DST;

        let keys = GpuVector::init(gpu.device(), &keys_inp, input_usages);
        let values = GpuVector::init(gpu.device(), &values_inp, input_usages);
        let out_keys = GpuVector::init(gpu.device(), &keys_inp, output_usages);
        let out_values = GpuVector::init(gpu.device(), &values_inp, output_usages);
        let staging_keys = GpuVector::init(gpu.device(), &keys_inp, staging_usage);
        let staging_values = GpuVector::init(gpu.device(), &values_inp, staging_usage);
        let num_points =
            GpuScalar::init(gpu.device(), keys_inp.len() as u32, BufferUsages::STORAGE);

        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        let mut pass = encoder.compute_pass("test", None);
        sort.dispatch(
            gpu.device(),
            &mut pass,
            &mut workspace,
            &keys,
            &values,
            &num_points,
            32,
            &out_keys,
            &out_values,
        );
        drop(pass);
        staging_keys.copy_from(&mut encoder, &out_keys);
        staging_values.copy_from(&mut encoder, &out_values);
        gpu.queue().submit(Some(encoder.finish()));

        let result_keys = staging_keys.read(gpu.device()).await.unwrap();
        let result_values = staging_values.read(gpu.device()).await.unwrap();

        let inds = cpu_argsort(&keys_inp);
        let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

        assert_eq!(DVector::from(ref_keys), DVector::from(result_keys));
        assert_eq!(DVector::from(ref_values), DVector::from(result_values));
    }
}
