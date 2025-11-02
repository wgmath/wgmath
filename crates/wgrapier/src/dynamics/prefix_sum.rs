//! GPU parallel prefix sum (scan) algorithm.
//!
//! This module implements an efficient parallel prefix sum on the GPU using a work-efficient
//! algorithm. Prefix sum is a fundamental parallel primitive used throughout the physics engine.
//!
//! # What is Prefix Sum?
//!
//! Given an input array `[a₀, a₁, a₂, ..., aₙ]`, the prefix sum produces:
//! `[0, a₀, a₀+a₁, a₀+a₁+a₂, ..., a₀+a₁+...+aₙ₋₁]`
//!
//! Note the special variant used here: a 0 is prepended as the first element, which is useful
//! for computing array indices and offsets.

use nalgebra::DVector;
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

/// GPU shader for parallel prefix sum.
///
/// This shader implements a work-efficient parallel scan algorithm optimized for GPUs.
#[derive(Shader)]
#[shader(src = "prefix_sum.wgsl", composable = false)]
pub struct WgPrefixSum {
    /// Main prefix sum kernel (both up-sweep and down-sweep).
    prefix_sum: ComputePipeline,
    /// Kernel for adding partial sums from coarser levels.
    add_data_grp: ComputePipeline,
}

impl WgPrefixSum {
    const THREADS: u32 = 256;

    /// Dispatches the prefix sum algorithm on GPU data.
    ///
    /// # Parameters
    ///
    /// - `device`: The WebGPU device
    /// - `pass`: Active compute pass
    /// - `workspace`: Workspace containing auxiliary buffers (resized automatically if needed)
    /// - `data`: Input/output buffer (modified in-place)
    ///
    /// # Panics
    ///
    /// Panics if `THREADS` is not 256, as the shared memory size is hardcoded in the shader.
    pub fn dispatch(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        workspace: &mut PrefixSumWorkspace,
        data: &GpuVector<u32>,
    ) {
        // If this assert fails, the kernel launches bellow must be changed because we are using
        // a fixed size for the shared memory currently.
        assert_eq!(
            Self::THREADS,
            256,
            "Internal error: prefix sum assumes a thread count equal to 256"
        );

        workspace.reserve(device, data.len() as u32);

        let ngroups0 = workspace.stages[0].buffer.len() as u32;
        let aux0 = &workspace.stages[0].buffer;
        KernelDispatch::new(device, pass, &self.prefix_sum)
            .bind0([data.buffer(), aux0.buffer()])
            .dispatch(ngroups0);

        for i in 0..workspace.num_stages - 1 {
            let ngroups = workspace.stages[i + 1].buffer.len() as u32;
            let buf = workspace.stages[i].buffer.buffer();
            let aux = workspace.stages[i + 1].buffer.buffer();

            KernelDispatch::new(device, pass, &self.prefix_sum)
                .bind0([buf, aux])
                .dispatch(ngroups);
        }

        if workspace.num_stages > 2 {
            for i in (0..workspace.num_stages - 2).rev() {
                let ngroups = workspace.stages[i + 1].buffer.len() as u32;
                let buf = workspace.stages[i].buffer.buffer();
                let aux = workspace.stages[i + 1].buffer.buffer();

                KernelDispatch::new(device, pass, &self.add_data_grp)
                    .bind0([buf, aux])
                    .dispatch(ngroups);
            }
        }

        if workspace.num_stages > 1 {
            KernelDispatch::new(device, pass, &self.add_data_grp)
                .bind0([data.buffer(), aux0.buffer()])
                .dispatch(ngroups0);
        }
    }

    /// CPU reference implementation of the prefix sum algorithm.
    ///
    /// This method computes the same result as the GPU version but on the CPU.
    /// Useful for testing and verification.
    ///
    /// # Parameters
    ///
    /// - `v`: Input/output vector (modified in-place)
    pub fn eval_cpu(&self, v: &mut DVector<u32>) {
        for i in 0..v.len() - 1 {
            v[i + 1] += v[i];
        }

        // NOTE: we actually have a special variant of the prefix-sum
        //       where the result is as if a 0 was appendend to the input vector.
        for i in (1..v.len()).rev() {
            v[i] = v[i - 1];
        }

        v[0] = 0;
    }
}

/// One stage in the multi-level prefix sum hierarchy.
struct PrefixSumStage {
    /// Maximum number of elements this stage can handle.
    capacity: u32,
    /// GPU buffer for storing partial sums at this level.
    buffer: GpuVector<u32>,
}

/// Workspace containing auxiliary buffers for hierarchical prefix sum.
///
/// The workspace maintains a hierarchy of buffers for the multi-level scan algorithm.
/// It automatically resizes when the input data size changes.
#[derive(Default)]
pub struct PrefixSumWorkspace {
    stages: Vec<PrefixSumStage>,
    num_stages: usize,
}

impl PrefixSumWorkspace {
    /// Creates a new empty workspace.
    pub fn new() -> Self {
        Self {
            stages: vec![],
            num_stages: 0,
        }
    }

    /// Creates a workspace pre-allocated for a specific buffer size.
    ///
    /// # Parameters
    ///
    /// - `device`: The WebGPU device for allocating buffers
    /// - `buffer_len`: Size of the data buffer that will be scanned
    pub fn with_capacity(device: &Device, buffer_len: u32) -> Self {
        let mut result = Self {
            stages: vec![],
            num_stages: 0,
        };
        result.reserve(device, buffer_len);
        result
    }

    /// Ensures the workspace has sufficient capacity for a given buffer size.
    ///
    /// Resizes auxiliary buffers if needed. This is called automatically by [`WgPrefixSum::dispatch`].
    ///
    /// # Parameters
    ///
    /// - `device`: The WebGPU device for allocating buffers
    /// - `buffer_len`: Size of the data buffer that will be scanned
    pub fn reserve(&mut self, device: &Device, buffer_len: u32) {
        let mut stage_len = buffer_len.div_ceil(WgPrefixSum::THREADS);

        if self.stages.is_empty() || self.stages[0].capacity < stage_len {
            // Reinitialize the auxiliary buffers.
            self.stages.clear();

            while stage_len != 1 {
                let buffer = GpuVector::init(
                    device,
                    DVector::<u32>::zeros(stage_len as usize),
                    BufferUsages::STORAGE,
                );
                self.stages.push(PrefixSumStage {
                    capacity: stage_len,
                    buffer,
                });

                stage_len = stage_len.div_ceil(WgPrefixSum::THREADS);
            }

            // The last stage always has only 1 element.
            self.stages.push(PrefixSumStage {
                capacity: 1,
                buffer: GpuVector::init(device, DVector::<u32>::zeros(1), BufferUsages::STORAGE),
            });
            self.num_stages = self.stages.len();
        } else if self.stages[0].buffer.len() as u32 != stage_len {
            // The stages have big enough buffers, but we need to adjust their length.
            self.num_stages = 0;
            while stage_len != 1 {
                self.num_stages += 1;
                stage_len = stage_len.div_ceil(WgPrefixSum::THREADS);
            }

            // The last stage always has only 1 element.
            self.num_stages += 1;
        }
    }

    /*
    pub fn read_max_scan_value(&mut self) -> cust::error::CudaResult<u32> {
        for stage in &self.stages {
            if stage.len == 1 {
                // This is the last stage, it contains the total sum.
                let mut value = [0u32];
                stage.buffer.index(0).copy_to(&mut value)?;
                return Ok(value[0]);
            }
        }

        panic!("The GPU prefix sum has not been initialized yet.")
    }
    */
}

#[cfg(test)]
mod test {
    use super::{PrefixSumWorkspace, WgPrefixSum};
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::CommandEncoderExt;
    use wgcore::tensor::GpuVector;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_prefix_sum() {
        const LEN: u32 = 15071;

        let gpu = GpuInstance::new().await.unwrap();
        let prefix_sum = WgPrefixSum::from_device(gpu.device()).unwrap();

        let inputs = vec![
            DVector::<u32>::from_fn(LEN as usize, |_, _| 1),
            DVector::<u32>::from_fn(LEN as usize, |i, _| i as u32),
            DVector::<u32>::new_random(LEN as usize).map(|e| e % 10_000),
        ];

        for v_cpu in inputs {
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            let v_gpu = GpuVector::init(
                gpu.device(),
                &v_cpu,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            );
            let staging = GpuVector::uninit(
                gpu.device(),
                v_cpu.len() as u32,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            );

            let mut workspace = PrefixSumWorkspace::with_capacity(gpu.device(), v_cpu.len() as u32);
            let mut pass = encoder.compute_pass("test", None);
            prefix_sum.dispatch(gpu.device(), &mut pass, &mut workspace, &v_gpu);
            drop(pass);
            staging.copy_from(&mut encoder, &v_gpu);

            let t0 = web_time::Instant::now();
            gpu.queue().submit(Some(encoder.finish()));
            let gpu_result = staging.read(gpu.device()).await.unwrap();
            println!("Gpu time: {}", t0.elapsed().as_secs_f32());

            let mut cpu_result = v_cpu.clone();

            let t0 = web_time::Instant::now();
            prefix_sum.eval_cpu(&mut cpu_result);
            println!("Cpu time: {}", t0.elapsed().as_secs_f32());
            // println!("input: {:?}", v_cpu);
            // println!("cpu output: {:?}", cpu_result);
            // println!("gpu output: {:?}", gpu_result);

            assert_eq!(DVector::from(gpu_result), cpu_result);
        }
    }
}
