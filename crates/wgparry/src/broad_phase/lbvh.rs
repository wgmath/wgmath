use crate::bounding_volumes::WgAabb;
use crate::math::{GpuSim, Point, Vector};
use crate::shapes::{GpuShape, WgShape};
use crate::utils::{RadixSort, RadixSortWorkspace};
use crate::{dim_shader_defs, substitute_aliases};
use naga_oil::compose::ComposerError;
use parry::bounding_volume::Aabb;
use wgcore::indirect::{DispatchIndirectArgs, WgIndirect};

#[cfg(feature = "dim2")]
use nalgebra::Vector2;
#[cfg(feature = "dim3")]
use nalgebra::Vector4;
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::Shader;
use wgebra::WgSim3;
use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgShape, WgAabb, WgIndirect),
    src = "./lbvh.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs",
    composable = false
)]
/// GPU shader for Linear Bounding Volume Hierarchy (LBVH) construction and traversal.
///
/// Implements the Karras 2012 parallel LBVH construction algorithm on the GPU, providing
/// O(n log n) collision detection suitable for large dynamic scenes.
pub struct WgLbvh {
    reset_collision_pairs: ComputePipeline,
    compute_domain: ComputePipeline,
    compute_morton: ComputePipeline,
    build: ComputePipeline,
    refit_leaves: ComputePipeline,
    refit_internal: ComputePipeline,
    #[allow(dead_code)]
    refit: ComputePipeline,
    find_collision_pairs: ComputePipeline,
    init_indirect_args: ComputePipeline,
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C)]
/// A node in the LBVH binary tree structure.
///
/// Each node represents either:
/// - **Leaf node**: Contains a single collider (when `left == right`)
/// - **Internal node**: Contains references to two child nodes
///
/// The tree is built using Morton codes for spatial sorting, enabling cache-friendly
/// traversal and efficient parallel construction.
pub struct LbvhNode {
    #[cfg(feature = "dim3")]
    aabb_mins: Vector4<f32>,
    #[cfg(feature = "dim3")]
    aabb_maxs: Vector4<f32>,
    #[cfg(feature = "dim2")]
    aabb_mins: Vector2<f32>,
    #[cfg(feature = "dim2")]
    aabb_maxs: Vector2<f32>,
    left: u32,
    right: u32,
    parent: u32,
    refit_count: u32,
}

impl LbvhNode {
    /// Extracts the AABB (axis-aligned bounding box) from this node.
    ///
    /// Returns a [`parry::bounding_volume::Aabb`] constructed from the node's min/max bounds.
    #[cfg(feature = "dim3")]
    pub fn aabb(&self) -> Aabb {
        Aabb::new(self.aabb_mins.xyz().into(), self.aabb_maxs.xyz().into())
    }

    /// Extracts the AABB (axis-aligned bounding box) from this node.
    ///
    /// Returns a [`parry::bounding_volume::Aabb`] constructed from the node's min/max bounds.
    #[cfg(feature = "dim2")]
    pub fn aabb(&self) -> Aabb {
        Aabb::new(self.aabb_mins.into(), self.aabb_maxs.into())
    }
}

/// GPU-resident state for LBVH construction and queries.
///
/// Maintains all GPU buffers needed for building and querying the LBVH:
/// - Morton codes and their sorted versions
/// - Collider indices (sorted by Morton code)
/// - The BVH tree structure itself
/// - Radix sort workspace for Morton code sorting
///
/// Buffers automatically resize when the number of colliders changes.
pub struct LbvhState {
    buffer_usages: BufferUsages, // Just for debugging if we want COPY_SRC
    #[cfg(feature = "dim3")]
    domain_aabb: GpuScalar<[Vector4<f32>; 2]>,
    #[cfg(feature = "dim2")]
    domain_aabb: GpuScalar<[Vector2<f32>; 2]>,
    n_sort: GpuScalar<u32>,
    unsorted_morton_keys: GpuVector<u32>,
    sorted_morton_keys: GpuVector<u32>,
    unsorted_colliders: GpuVector<u32>,
    sorted_colliders: GpuVector<u32>,
    tree: GpuVector<LbvhNode>,
    sort_workspace: RadixSortWorkspace,
}

/// High-level LBVH broad-phase interface (shaders only).
///
/// Provides the complete LBVH pipeline:
/// 1. Compute AABBs and domain bounds
/// 2. Generate Morton codes for spatial sorting
/// 3. Sort colliders by Morton code
/// 4. Build binary tree structure
/// 5. Traverse tree to find collision pairs
pub struct Lbvh {
    shaders: WgLbvh,
    sort: RadixSort,
}

impl LbvhState {
    /// Creates a new LBVH state with default buffer usage flags.
    ///
    /// Initializes all buffers with `BufferUsages::STORAGE` flag for compute shader access.
    pub fn new(device: &Device) -> Result<Self, ComposerError> {
        Self::with_usages(device, BufferUsages::STORAGE)
    }

    /// Creates a new LBVH state with custom buffer usage flags.
    ///
    /// Allows specifying custom usage flags for debugging or special use cases
    /// (e.g., adding `COPY_SRC` for buffer readback).
    pub fn with_usages(device: &Device, usages: BufferUsages) -> Result<Self, ComposerError> {
        Ok(Self {
            n_sort: GpuScalar::init(device, 0, usages),
            domain_aabb: GpuScalar::uninit(device, usages),
            unsorted_morton_keys: GpuVector::uninit(device, 0, usages),
            sorted_morton_keys: GpuVector::uninit(device, 0, usages),
            unsorted_colliders: GpuVector::uninit(device, 0, usages),
            sorted_colliders: GpuVector::uninit(device, 0, usages),
            tree: GpuVector::uninit(device, 0, usages),
            sort_workspace: RadixSortWorkspace::new(device),
            buffer_usages: usages,
        })
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
        }
    }
}

impl Lbvh {
    const WORKGROUP_SIZE: u32 = 64;

    /// Creates a new LBVH instance by compiling shaders on the given device.
    ///
    /// # Errors
    ///
    /// Returns an error if shader compilation fails.
    pub fn from_device(device: &Device) -> Result<Self, ComposerError> {
        Ok(Self {
            shaders: WgLbvh::from_device(device)?,
            sort: RadixSort::from_device(device)?,
        })
    }

    /// Rebuilds the LBVH tree from current collider poses and shapes.
    ///
    /// This method:
    /// 1. Computes AABBs for all colliders
    /// 2. Calculates the bounding domain
    /// 3. Generates Morton codes for spatial sorting
    /// 4. Sorts colliders by Morton code using radix sort
    /// 5. Builds the binary BVH tree structure
    ///
    /// Should be called each frame before [`find_pairs`](Self::find_pairs) if colliders have moved.
    ///
    /// # Parameters
    ///
    /// - `device`: The GPU device
    /// - `pass`: Active compute pass to record commands into
    /// - `state`: Mutable LBVH state (buffers may be resized if needed)
    /// - `colliders_len`: Number of colliders to process
    /// - `poses`: Collider world-space poses
    /// - `shapes`: Collider shapes
    /// - `num_shapes`: Scalar buffer containing the collider count
    pub fn update_tree(
        &self,
        device: &Device,
        pass: &mut ComputePass<'_>,
        state: &mut LbvhState,
        colliders_len: u32,
        poses: &GpuVector<GpuSim>,
        vertex_buffers: &GpuVector<Point<f32>>,
        index_buffers: &GpuVector<u32>,
        shapes: &GpuVector<GpuShape>,
        num_shapes: &GpuScalar<u32>,
    ) {
        state.resize_buffers(device, colliders_len);

        // Bind group 0.
        let num_colliders = (num_shapes.buffer(), 0);
        let poses = (poses.buffer(), 1);
        let shapes = (shapes.buffer(), 2);
        let domain_aabb = (state.domain_aabb.buffer(), 6);
        let unsorted_morton_keys = (state.unsorted_morton_keys.buffer(), 7);
        let sorted_morton_keys = (state.sorted_morton_keys.buffer(), 7);
        let sorted_colliders = (state.sorted_colliders.buffer(), 8);
        let tree = (state.tree.buffer(), 9);

        let vertices = (vertex_buffers.buffer(), 0);

        // Dispatch everything.
        KernelDispatch::new(device, pass, &self.shaders.compute_domain)
            .bind_at(0, [num_colliders, poses, domain_aabb])
            .dispatch(1);

        KernelDispatch::new(device, pass, &self.shaders.compute_morton)
            .bind_at(0, [num_colliders, poses, domain_aabb, unsorted_morton_keys])
            .dispatch(colliders_len.div_ceil(Self::WORKGROUP_SIZE));

        self.sort.dispatch(
            device,
            pass,
            &mut state.sort_workspace,
            &state.unsorted_morton_keys,
            &state.unsorted_colliders,
            &state.n_sort,
            32,
            &state.sorted_morton_keys,
            &state.sorted_colliders,
        );

        KernelDispatch::new(device, pass, &self.shaders.build)
            .bind_at(0, [num_colliders, sorted_morton_keys, tree])
            .dispatch((colliders_len - 1).div_ceil(Self::WORKGROUP_SIZE));

        KernelDispatch::new(device, pass, &self.shaders.refit_leaves)
            .bind_at(0, [num_colliders, tree, poses, shapes, sorted_colliders])
            .bind_at(1, [])
            .bind_at(2, [vertices])
            .dispatch(colliders_len.div_ceil(Self::WORKGROUP_SIZE));

        KernelDispatch::new(device, pass, &self.shaders.refit_internal)
            .bind_at(0, [num_colliders, tree])
            .dispatch(1);
        // .dispatch(colliders_len.div_ceil(Self::WORKGROUP_SIZE));

        // KernelDispatch::new(device, pass, &self.shaders.refit)
        //     .bind_at(0, [num_colliders, tree, poses, shapes, sorted_colliders])
        //     .dispatch(colliders_len.div_ceil(Self::WORKGROUP_SIZE));
    }

    /// Traverses the LBVH tree to find potentially colliding pairs.
    ///
    /// After the tree has been built with [`update_tree`](Self::update_tree), this method
    /// traverses it to identify pairs of colliders whose AABBs overlap.
    ///
    /// # Parameters
    ///
    /// - `device`: The GPU device
    /// - `pass`: Active compute pass to record commands into
    /// - `state`: LBVH state containing the built tree
    /// - `colliders_len`: Number of colliders in the scene
    /// - `num_shapes`: Scalar buffer containing the collider count
    /// - `collision_pairs`: Output buffer for potentially colliding pairs
    /// - `collision_pairs_len`: Output count of collision pairs found
    /// - `collision_pairs_indirect`: Indirect dispatch args for subsequent kernels
    pub fn find_pairs(
        &self,
        device: &Device,
        pass: &mut ComputePass<'_>,
        state: &mut LbvhState,
        colliders_len: u32,
        num_shapes: &GpuScalar<u32>,
        collision_pairs: &GpuVector<[u32; 2]>,
        collision_pairs_len: &GpuScalar<u32>,
        collision_pairs_indirect: &GpuScalar<DispatchIndirectArgs>,
    ) {
        // Bind group 0.
        let num_colliders = (num_shapes.buffer(), 0);
        let collision_pairs = (collision_pairs.buffer(), 3);
        let collision_pairs_len = (collision_pairs_len.buffer(), 4);
        let collision_pairs_indirect = (collision_pairs_indirect.buffer(), 5);
        let tree = (state.tree.buffer(), 9);

        KernelDispatch::new(device, pass, &self.shaders.reset_collision_pairs)
            .bind_at(0, [collision_pairs_len])
            .dispatch(1);

        KernelDispatch::new(device, pass, &self.shaders.find_collision_pairs)
            .bind_at(
                0,
                [num_colliders, collision_pairs_len, collision_pairs, tree],
            )
            .dispatch(colliders_len.div_ceil(Self::WORKGROUP_SIZE));

        KernelDispatch::new(device, pass, &self.shaders.init_indirect_args)
            .bind_at(0, [collision_pairs_len, collision_pairs_indirect])
            .dispatch(1);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use parry::bounding_volume::BoundingVolume;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::CommandEncoderExt;

    #[cfg(feature = "dim2")]
    use na::{Similarity2, Vector2};
    #[cfg(feature = "dim3")]
    use na::{Similarity3, Vector3};

    #[futures_test::test]
    #[serial_test::serial]
    async fn tree_construction() {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let gpu = GpuInstance::new().await.unwrap();
        let lbvh = Lbvh::from_device(gpu.device()).unwrap();
        let mut state = LbvhState::with_usages(gpu.device(), storage).unwrap();
        const LEN: u32 = 1000;
        let poses: Vec<_> = (0..LEN)
            .map(|i| {
                #[cfg(feature = "dim3")]
                {
                    Similarity3::new(
                        -Vector3::new(i as f32, (i as f32).sin(), (i as f32).cos()),
                        na::zero(),
                        1.0,
                    )
                }
                #[cfg(feature = "dim2")]
                {
                    Similarity2::new(-Vector2::new(i as f32, (i as f32).sin()), 0.0, 1.0)
                }
            })
            .collect();
        #[cfg(feature = "dim3")]
        let gpu_poses_data: Vec<GpuSim> = poses.clone();
        #[cfg(feature = "dim2")]
        let gpu_poses_data: Vec<GpuSim> = poses.iter().map(|p| (*p).into()).collect();
        let shapes: Vec<_> = vec![GpuShape::ball(0.5); LEN as usize];

        let gpu_vertices = GpuVector::encase(gpu.device(), &[], storage);
        let gpu_indices = GpuVector::init(gpu.device(), &[], storage);
        let gpu_poses = GpuVector::init(gpu.device(), &gpu_poses_data, storage);
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
        lbvh.update_tree(
            gpu.device(),
            &mut pass,
            &mut state,
            LEN,
            &gpu_poses,
            &gpu_vertices,
            &gpu_indices,
            &gpu_shapes,
            &gpu_num_shapes,
        );
        lbvh.find_pairs(
            gpu.device(),
            &mut pass,
            &mut state,
            LEN,
            &gpu_num_shapes,
            &gpu_collision_pairs,
            &gpu_collision_pairs_len,
            &gpu_collision_pairs_indirect,
        );
        drop(pass);
        gpu.queue().submit(Some(encoder.finish()));

        // Check result of `compute_domain`.
        let domain = state.domain_aabb.slow_read(&gpu).await[0];
        let pts: Vec<_> = poses
            .iter()
            .map(|p| p.isometry.translation.vector.into())
            .collect();
        let domain_cpu = Aabb::from_points(pts.iter().copied());
        #[cfg(feature = "dim3")]
        {
            assert_eq!(domain_cpu.mins.coords, domain[0].xyz());
            assert_eq!(domain_cpu.maxs.coords, domain[1].xyz());
        }
        #[cfg(feature = "dim2")]
        {
            assert_eq!(domain_cpu.mins.coords, domain[0].xy());
            assert_eq!(domain_cpu.maxs.coords, domain[1].xy());
        }

        // Check result of `compute_morton`.
        let mortons = state.unsorted_morton_keys.slow_read(&gpu).await;
        let morton_cpu: Vec<_> = pts
            .iter()
            .map(|pt| {
                let normalized = (pt - domain_cpu.mins).component_div(&domain_cpu.extents());
                #[cfg(feature = "dim3")]
                {
                    morton(normalized.x, normalized.y, normalized.z)
                }
                #[cfg(feature = "dim2")]
                {
                    morton(normalized.x, normalized.y)
                }
            })
            .collect();
        // Check morton codes match (allow small differences due to floating point precision).
        for (i, (&cpu, &gpu)) in morton_cpu.iter().zip(mortons.iter()).enumerate() {
            let diff = cpu.abs_diff(gpu);
            assert!(
                diff <= 2,
                "Morton code mismatch at index {}: CPU={}, GPU={}, diff={}",
                i,
                cpu,
                gpu,
                diff
            );
        }

        // Check result of `sort`.
        // Use GPU morton values for sorting to ensure exact match.
        let mut sorted_colliders_cpu: Vec<_> = (0..LEN).collect();
        sorted_colliders_cpu.sort_by_key(|i| mortons[*i as usize]);
        let mut morton_sorted = mortons.to_vec();
        morton_sorted.sort();
        let sorted_mortons = state.sorted_morton_keys.slow_read(&gpu).await;
        let sorted_colliders = state.sorted_colliders.slow_read(&gpu).await;
        assert_eq!(sorted_mortons, morton_sorted);
        assert_eq!(sorted_colliders, sorted_colliders_cpu);

        // Check result of `build`.
        let tree = state.tree.slow_read(&gpu).await;

        {
            // Check that a traversal covers all the nodes and that there is no loop.
            let mut visited = vec![false; tree.len()];
            let mut stack = vec![0];
            while let Some(curr) = stack.pop() {
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
                        parry::math::Vector::repeat(0.5)
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

    #[cfg(feature = "dim3")]
    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
    fn expand_bits(v: u32) -> u32 {
        let mut vv = v.wrapping_mul(0x00010001) & 0xFF0000FF;
        vv = vv.wrapping_mul(0x00000101) & 0x0F00F00F;
        vv = vv.wrapping_mul(0x00000011) & 0xC30C30C3;
        vv = vv.wrapping_mul(0x00000005) & 0x49249249;
        vv
    }

    #[cfg(feature = "dim3")]
    // Calculates a 30-bit Morton code for the
    // given 3D point located within the unit cube [0,1].
    fn morton(x: f32, y: f32, z: f32) -> u32 {
        let scaled_x = (x * 1024.0).clamp(0.0, 1023.0);
        let scaled_y = (y * 1024.0).clamp(0.0, 1023.0);
        let scaled_z = (z * 1024.0).clamp(0.0, 1023.0);
        let xx = expand_bits(scaled_x as u32);
        let yy = expand_bits(scaled_y as u32);
        let zz = expand_bits(scaled_z as u32);
        xx * 4 + yy * 2 + zz
    }

    #[cfg(feature = "dim2")]
    // Expands a 16-bit integer into 32 bits
    // by inserting 1 zero after each bit.
    fn expand_bits(v: u32) -> u32 {
        let mut x = v & 0x0000ffff;
        x = (x | (x << 8)) & 0x00ff00ff;
        x = (x | (x << 4)) & 0x0f0f0f0f;
        x = (x | (x << 2)) & 0x33333333;
        x = (x | (x << 1)) & 0x55555555;
        x
    }

    #[cfg(feature = "dim2")]
    // Calculates a 32-bit Morton code for the
    // given 2D point located within the unit square [0,1].
    fn morton(x: f32, y: f32) -> u32 {
        let scaled_x = (x * 65536.0).clamp(0.0, 65535.0);
        let scaled_y = (y * 65536.0).clamp(0.0, 65535.0);
        let xx = expand_bits(scaled_x as u32);
        let yy = expand_bits(scaled_y as u32);
        xx | (yy << 1)
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
