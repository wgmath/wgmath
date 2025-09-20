use crate::bounding_volumes::WgAabb;
use crate::math::{GpuSim, Vector};
use crate::shapes::{GpuShape, WgShape};
use crate::substitute_aliases;
use crate::utils::{RadixSort, RadixSortWorkspace};
use encase::ShaderType;
use na::Vector4;
use naga_oil::compose::ComposerError;
use parry::bounding_volume::Aabb;
use wgcore::indirect::{DispatchIndirectArgs, WgIndirect};
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::{test_shader_compilation, Shader};
use wgebra::WgSim3;
use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgShape, WgAabb, WgIndirect),
    src = "./lbvh.wgsl",
    src_fn = "substitute_aliases",
    composable = false
)]
pub struct WgLbvh {
    reset_collision_pairs: ComputePipeline,
    compute_domain: ComputePipeline,
    compute_morton: ComputePipeline,
    build: ComputePipeline,
    refit_leaves: ComputePipeline,
    refit_internal: ComputePipeline,
    refit: ComputePipeline,
    find_collision_pairs: ComputePipeline,
    init_indirect_args: ComputePipeline,
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C)]
pub struct LbvhNode {
    aabb_mins: Vector4<f32>,
    aabb_maxs: Vector4<f32>,
    left: u32,
    right: u32,
    parent: u32,
    refit_count: u32,
}

impl LbvhNode {
    pub fn aabb(&self) -> Aabb {
        Aabb::new(self.aabb_mins.xyz().into(), self.aabb_maxs.xyz().into())
    }
}

pub struct Lbvh {
    shaders: WgLbvh,
    buffer_usages: BufferUsages, // Just for debugging if we want COPY_SRC
    domain_aabb: GpuScalar<[Vector4<f32>; 2]>,
    n_sort: GpuScalar<u32>,
    unsorted_morton_keys: GpuVector<u32>,
    sorted_morton_keys: GpuVector<u32>,
    unsorted_colliders: GpuVector<u32>,
    sorted_colliders: GpuVector<u32>,
    tree: GpuVector<LbvhNode>,
    sort: RadixSort,
    sort_workspace: RadixSortWorkspace,
}

impl Lbvh {
    pub fn new(device: &Device) -> Result<Self, ComposerError> {
        Self::with_usages(device, BufferUsages::STORAGE)
    }
    pub fn with_usages(device: &Device, usages: BufferUsages) -> Result<Self, ComposerError> {
        Ok(Self {
            shaders: WgLbvh::from_device(device)?,
            n_sort: GpuScalar::init(device, 0, usages),
            domain_aabb: GpuScalar::uninit(device, usages),
            unsorted_morton_keys: GpuVector::uninit(device, 0, usages),
            sorted_morton_keys: GpuVector::uninit(device, 0, usages),
            unsorted_colliders: GpuVector::uninit(device, 0, usages),
            sorted_colliders: GpuVector::uninit(device, 0, usages),
            tree: GpuVector::uninit(device, 0, usages),
            sort: RadixSort::from_device(device)?,
            sort_workspace: RadixSortWorkspace::new(device),
            buffer_usages: usages,
        })
    }

    pub fn launch(
        &mut self,
        device: &Device,
        pass: &mut ComputePass<'_>,
        colliders_len: u32,
        poses: &GpuVector<GpuSim>,
        shapes: &GpuVector<GpuShape>,
        num_shapes: &GpuScalar<u32>,
        collision_pairs: &GpuVector<[u32; 2]>,
        collision_pairs_len: &GpuScalar<u32>,
        collision_pairs_indirect: &GpuScalar<DispatchIndirectArgs>,
    ) {
        const WORKGROUP_SIZE: u32 = 64;

        self.resize_buffers(device, colliders_len);

        // Bind group 0.
        let num_colliders = (num_shapes.buffer(), 0);
        let poses = (poses.buffer(), 1);
        let shapes = (shapes.buffer(), 2);
        let collision_pairs = (collision_pairs.buffer(), 3);
        let collision_pairs_len = (collision_pairs_len.buffer(), 4);
        let collision_pairs_indirect = (collision_pairs_indirect.buffer(), 5);
        let domain_aabb = (self.domain_aabb.buffer(), 6);
        let unsorted_morton_keys = (self.unsorted_morton_keys.buffer(), 7);
        let sorted_morton_keys = (self.sorted_morton_keys.buffer(), 7);
        let sorted_colliders = (self.sorted_colliders.buffer(), 8);
        let tree = (self.tree.buffer(), 9);

        // Dispatch everything.
        KernelDispatch::new(device, pass, &self.shaders.reset_collision_pairs)
            .bind_at(0, [collision_pairs_len])
            .dispatch(1);

        KernelDispatch::new(device, pass, &self.shaders.compute_domain)
            .bind_at(0, [num_colliders, poses, domain_aabb])
            .dispatch(1);

        KernelDispatch::new(device, pass, &self.shaders.compute_morton)
            .bind_at(0, [num_colliders, poses, domain_aabb, unsorted_morton_keys])
            .dispatch(colliders_len.div_ceil(WORKGROUP_SIZE));

        self.sort.dispatch(
            device,
            pass,
            &mut self.sort_workspace,
            &self.unsorted_morton_keys,
            &self.unsorted_colliders,
            &self.n_sort,
            32,
            &self.sorted_morton_keys,
            &self.sorted_colliders,
        );

        KernelDispatch::new(device, pass, &self.shaders.build)
            .bind_at(0, [num_colliders, sorted_morton_keys, tree])
            .dispatch((colliders_len - 1).div_ceil(WORKGROUP_SIZE));

        KernelDispatch::new(device, pass, &self.shaders.refit_leaves)
            .bind_at(0, [num_colliders, tree, poses, shapes, sorted_colliders])
            .dispatch(colliders_len.div_ceil(WORKGROUP_SIZE));

        KernelDispatch::new(device, pass, &self.shaders.refit_internal)
            .bind_at(0, [num_colliders, tree])
            .dispatch(1);

        // KernelDispatch::new(device, pass, &self.shaders.refit)
        //     .bind_at(0, [num_colliders, tree, poses, shapes, sorted_colliders])
        //     .dispatch(colliders_len.div_ceil(WORKGROUP_SIZE));

        KernelDispatch::new(device, pass, &self.shaders.find_collision_pairs)
            .bind_at(
                0,
                [num_colliders, collision_pairs_len, collision_pairs, tree],
            )
            .dispatch(colliders_len.div_ceil(WORKGROUP_SIZE));

        KernelDispatch::new(device, pass, &self.shaders.init_indirect_args)
            .bind_at(0, [collision_pairs_len, collision_pairs_indirect])
            .dispatch(1);
    }

    fn resize_buffers(&mut self, device: &Device, colliders_len: u32) {
        if self.tree.len() < 2 * colliders_len as u64 {
            self.unsorted_morton_keys =
                GpuVector::uninit(device, colliders_len, self.buffer_usages);
            self.sorted_morton_keys = GpuVector::uninit(device, colliders_len, self.buffer_usages);
            let unsorted_colliders: Vec<_> = (0..colliders_len).collect();
            self.unsorted_colliders =
                GpuVector::init(device, &unsorted_colliders, self.buffer_usages);
            self.sorted_colliders = GpuVector::uninit(device, colliders_len, self.buffer_usages);
            self.tree = GpuVector::uninit(device, 2 * colliders_len, self.buffer_usages);

            // FIXME: we should instead write the len into the existing buffer at each frame
            //        to handle dynamic body/collider insertion/removal.
            self.n_sort = GpuScalar::init(device, colliders_len, self.buffer_usages);
            println!("> nuw bufs");
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::Isometry;
    use na::Similarity3;
    use parry::bounding_volume::BoundingVolume;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::CommandEncoderExt;

    #[futures_test::test]
    #[serial_test::serial]
    async fn tree_construction() {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let gpu = GpuInstance::new().await.unwrap();
        let mut lbvh = Lbvh::with_usages(gpu.device(), storage).unwrap();
        const LEN: u32 = 1000;
        let poses: Vec<_> = (0..LEN)
            .map(|i| {
                Similarity3::new(
                    -Vector::new(i as f32, (i as f32).sin(), (i as f32).cos()),
                    na::zero(),
                    1.0,
                )
            })
            .collect();
        let shapes: Vec<_> = vec![GpuShape::ball(0.5); LEN as usize];

        let gpu_poses = GpuVector::init(gpu.device(), &poses, storage);
        let gpu_shapes = GpuVector::init(gpu.device(), &shapes, storage);
        let gpu_num_shapes = GpuScalar::init(
            gpu.device(),
            LEN,
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        );
        let gpu_collision_pairs = GpuVector::uninit(gpu.device(), 10000, storage);
        let gpu_collision_pairs_len = GpuScalar::init(gpu.device(), 0, storage);
        let gpu_collision_pairs_indirect = GpuScalar::uninit(gpu.device(), storage);

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut pass = encoder.compute_pass("", None);
        lbvh.launch(
            gpu.device(),
            &mut pass,
            LEN,
            &gpu_poses,
            &gpu_shapes,
            &gpu_num_shapes,
            &gpu_collision_pairs,
            &gpu_collision_pairs_len,
            &gpu_collision_pairs_indirect,
        );
        drop(pass);
        gpu.queue().submit(Some(encoder.finish()));

        // Check result of `compute_domain`.
        let domain = lbvh.domain_aabb.slow_read(&gpu).await[0];
        let pts: Vec<_> = poses
            .iter()
            .map(|p| p.isometry.translation.vector.into())
            .collect();
        let domain_cpu = Aabb::from_points(&pts);
        assert_eq!(domain_cpu.mins.coords, domain[0].xyz());
        assert_eq!(domain_cpu.maxs.coords, domain[1].xyz());

        // Check result of `compute_morton`.
        let mortons = lbvh.unsorted_morton_keys.slow_read(&gpu).await;
        let mut morton_cpu: Vec<_> = pts
            .iter()
            .map(|pt| {
                let normalized = (pt - domain_cpu.mins).component_div(&domain_cpu.extents());
                morton(normalized.x, normalized.y, normalized.z)
            })
            .collect();
        assert_eq!(morton_cpu, mortons);

        // Check result of `sort`.
        let mut sorted_colliders_cpu: Vec<_> = (0..LEN).collect();
        sorted_colliders_cpu.sort_by_key(|i| morton_cpu[*i as usize]);
        morton_cpu.sort();
        let mut sorted_mortons = lbvh.sorted_morton_keys.slow_read(&gpu).await;
        let sorted_colliders = lbvh.sorted_colliders.slow_read(&gpu).await;
        assert_eq!(sorted_mortons, morton_cpu);
        assert_eq!(sorted_colliders, sorted_colliders_cpu);

        // Check result of `build`.
        let mut tree = lbvh.tree.slow_read(&gpu).await;

        {
            // Check that a traversal covers all the nodes and that there is no loop.
            let mut visited = vec![false; tree.len()];
            let mut stack = vec![0];
            let mut loops = 0;
            while let Some(curr) = stack.pop() {
                loops += 1;
                let node = &tree[curr];
                assert!(!visited[curr]);
                visited[curr] = true;
                if curr < LEN as usize - 1 {
                    // This is an internal node
                    stack.push(node.left as usize);
                    stack.push(node.right as usize);
                }
            }

            assert_eq!(visited.iter().filter(|e| **e).count(), LEN as usize * 2 - 1);

            // Check parent pointers.
            for (i, node) in tree[..LEN as usize - 1].iter().enumerate() {
                assert_eq!(tree[node.right as usize].parent, i as u32);
                assert_eq!(tree[node.left as usize].parent, i as u32);
            }
        }

        // Check result of `refit`.
        {
            // Check that the leaf AABBs are correct.
            let first_leaf_id = LEN - 1;
            for i in 0..LEN {
                let node = &tree[(first_leaf_id + i) as usize];
                let collider = sorted_colliders[i as usize];
                assert_eq!(
                    node.aabb(),
                    Aabb::from_half_extents(
                        poses[collider as usize].isometry.translation.vector.into(),
                        Vector::repeat(0.5)
                    )
                );
            }

            // Check that each AABB encloses the AABB of its children.
            for i in 0..LEN - 1 {
                let node = &tree[i as usize];
                let left = &tree[node.left as usize];
                let right = &tree[node.right as usize];
                println!("Testing: {} -> ({},{})", i, node.left, node.right);

                println!("Node: {:?}", node.aabb());
                println!("Left: {:?}", left.aabb());
                println!("Right: {:?}", right.aabb());
                assert_eq!(node.aabb(), left.aabb().merged(&right.aabb()));
            }
        }
    }

    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
    fn expandBits(v: u32) -> u32 {
        let mut vv = (v * 0x00010001) & 0xFF0000FF;
        vv = (vv * 0x00000101) & 0x0F00F00F;
        vv = (vv * 0x00000011) & 0xC30C30C3;
        vv = (vv * 0x00000005) & 0x49249249;
        vv
    }

    // Calculates a 30-bit Morton code for the
    // given 3D point located within the unit cube [0,1].
    fn morton(x: f32, y: f32, z: f32) -> u32 {
        let scaled_x = (x * 1024.0).max(0.0).min(1023.0);
        let scaled_y = (y * 1024.0).max(0.0).min(1023.0);
        let scaled_z = (z * 1024.0).max(0.0).min(1023.0);
        let xx = expandBits(scaled_x as u32);
        let yy = expandBits(scaled_y as u32);
        let zz = expandBits(scaled_z as u32);
        xx * 4 + yy * 2 + zz
    }
    //
    // struct PrefixLen<'a> {
    //     morton_keys: &'a [u32],
    //     num_colliders: usize,
    // }
    //
    // impl<'a> PrefixLen<'a> {
    //     fn morton_at(&self, i: i32) -> i32 {
    //         // TODO PERF: would it be meaningful to add sentinels at the begining
    //         //            and end of the morton_keys array so we don’t have to check
    //         //            bounds?
    //         if i < 0 || i > self.num_colliders as i32 - 1 {
    //             return -1;
    //         } else {
    //             return self.morton_keys[i as usize] as i32;
    //         }
    //     }
    //
    //     fn prefix_len(&self, curr_key: u32, other_index: i32) -> i32 {
    //         let other_key = self.morton_at(other_index);
    //         (curr_key as i32 ^ other_key).leading_zeros() as i32
    //     }
    // }
    //
    // /// Builds each node of the tree in parallel.
    // ///
    // /// This only computes the tree topology (children and parent pointers).
    // /// This doesn’t update the bounding boxes. Call `refit` for updating bounding boxes!
    // fn build(tree: &mut [LbvhNode], num_colliders: usize, morton_keys: &[u32]) {
    //     let num_internal_nodes = num_colliders - 1;
    //     let first_leaf_id = num_internal_nodes;
    //     let pl = PrefixLen {
    //         morton_keys,
    //         num_colliders,
    //     };
    //
    //     for i in 0..num_internal_nodes {
    //         // Determine the direction of the range (+1 or -1).
    //         let ii = i as i32;
    //         let curr_key = morton_keys[i];
    //         let d = (pl.prefix_len(curr_key, ii + 1) - pl.prefix_len(curr_key, ii - 1)).signum();
    //
    //         // Compute upper bound for the length of the range.
    //         let delta_min = pl.prefix_len(curr_key, ii - d);
    //         let mut l = 0;
    //         while pl.prefix_len(curr_key, ii + (l + 1) * d) > delta_min {
    //             l += 1;
    //         }
    //
    //         let mut lmax = 2; // TODO PERF: start at 128 ?
    //         while pl.prefix_len(curr_key, ii + lmax * d) > delta_min {
    //             lmax *= 2; // TODO PERF: multiply by 4 instead of 2 ?
    //         }
    //
    //         // Find the other end using binary search.
    //         let mut l = 0;
    //         let mut t = lmax / 2;
    //         while t >= 1 {
    //             if pl.prefix_len(curr_key, ii + (l + t) * d) > delta_min {
    //                 l += t;
    //             }
    //
    //             t /= 2;
    //         }
    //
    //         let j = ii + l * d;
    //
    //         // Find the split position using binary search.
    //         let delta_node = pl.prefix_len(curr_key, j);
    //
    //         let mut s = 0;
    //         while pl.prefix_len(curr_key, ii + (s + 1) * d) > delta_node {
    //             s += 1;
    //         }
    //         let seq_s = s;
    //
    //         let mut s = 0;
    //         let mut t = (l as u32).div_ceil(2) as i32;
    //         loop {
    //             if pl.prefix_len(curr_key, ii + (s + t) * d) > delta_node {
    //                 s += t;
    //             }
    //
    //             if t == 1 {
    //                 break;
    //             } else {
    //                 t = (t as u32).div_ceil(2) as i32;
    //             }
    //         }
    //
    //         println!(
    //             "base t: {}, delta: {delta_node}, plen seq: {}, plen: {}",
    //             l / 2,
    //             pl.prefix_len(curr_key, ii + seq_s * d),
    //             pl.prefix_len(curr_key, ii + s * d)
    //         );
    //         assert_eq!(seq_s, s);
    //
    //         let gamma = ii + s * d + d.min(0);
    //
    //         // Output child and parent pointers.
    //         let left = if ii.min(j) == gamma {
    //             first_leaf_id as i32 + gamma
    //         } else {
    //             gamma
    //         };
    //         let right = if ii.max(j) == gamma + 1 {
    //             first_leaf_id as i32 + gamma + 1
    //         } else {
    //             gamma + 1
    //         };
    //         tree[i].left = left as u32;
    //         tree[i].right = right as u32;
    //         tree[i].refit_count = 0; // This is a good opportunity to reset the `refit_count` too.
    //         tree[left as usize].parent = i as u32;
    //         tree[right as usize].parent = i as u32;
    //     }
    // }
}
