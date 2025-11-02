//! Utilities for initializing and slicing tensors, matrices, vectors, and scalars gpu storage
//! buffers.

use crate::gpu::GpuInstance;
use crate::shapes::ViewShape;
use bytemuck::Pod;
use encase::internal::{CreateFrom, ReadFrom, WriteInto};
use encase::{ShaderSize, ShaderType, StorageBuffer};
use nalgebra::{Dim, IsContiguous, Matrix, Storage};
use std::marker::PhantomData;
use std::mem::size_of;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, BufferView, CommandEncoder, Device,
};

#[derive(Copy, Clone)]
pub struct ColumnMajor;
#[derive(Copy, Clone)]
pub struct RowMajor;

pub trait MatrixOrdering: Copy + Clone {
    fn is_row_major() -> bool;
    fn is_column_major() -> bool {
        !Self::is_row_major()
    }
}

impl MatrixOrdering for ColumnMajor {
    fn is_row_major() -> bool {
        false
    }
}

impl MatrixOrdering for RowMajor {
    fn is_row_major() -> bool {
        true
    }
}

/// A storage buffer containing a single value.
pub type GpuScalar<T> = GpuTensor<T, 0>;
/// A storage buffer containing a vector.
pub type GpuVector<T> = GpuTensor<T, 1>;
/// A storage buffer containing a matrix.
pub type GpuMatrix<T> = GpuTensor<T, 2>;
/// A storage buffer containing a cube (order-3 tensor).
pub type GpuCube<T> = GpuTensor<T, 3>;

/// A view, over a storage buffer, containing a single value.
pub type GpuScalarView<'a, T, Ordering = ColumnMajor> = GpuTensorView<'a, T, Ordering, 0>;
/// A view, over a storage buffer, containing a vector.
pub type GpuVectorView<'a, T, Ordering = ColumnMajor> = GpuTensorView<'a, T, Ordering, 1>;
/// A view, over a storage buffer, containing a matrix.
pub type GpuMatrixView<'a, T, Ordering = ColumnMajor> = GpuTensorView<'a, T, Ordering, 2>;
/// A view, over a storage buffer, containing a cube (order-3 tensor).
pub type GpuCubeView<'a, T, Ordering = ColumnMajor> = GpuTensorView<'a, T, Ordering, 3>;

/// Helper struct for creating gpu storage buffers (scalars, vectors, matrices, tensors).
///
/// When building a scalar, vector, or matrix tensor, it might be more convenient to call
/// [`GpuScalar::init`], [`GpuVector::init`], [`GpuMatrix::init`] (or their `encase` variants:
/// [`GpuScalar::encase`], [`GpuVector::encase`], [`GpuMatrix::encase`]; or their uninitialized
/// variants [`GpuScalar::uninit`], [`GpuVector::uninit`], [`GpuMatrix::uninit`]).
pub struct TensorBuilder<const DIM: usize> {
    shape: [u32; DIM],
    usage: BufferUsages,
    label: Option<String>,
}

impl TensorBuilder<0> {
    /// Starts building a storage buffer containing a single scalar value.
    pub fn scalar(usage: BufferUsages) -> Self {
        Self::tensor([], usage)
    }
}

impl TensorBuilder<1> {
    /// Starts building a storage buffer containing a vector.
    pub fn vector(dim: u32, usage: BufferUsages) -> Self {
        Self::tensor([dim], usage)
    }
}

impl TensorBuilder<2> {
    /// Starts building a storage buffer containing a single matrix with `nrows` rows and
    /// `ncols` columns.
    pub fn matrix(nrows: u32, ncols: u32, usage: BufferUsages) -> Self {
        Self::tensor([nrows, ncols], usage)
    }
}

impl<const DIM: usize> TensorBuilder<DIM> {
    /// Starts building a storage buffer containing a tensor with the specified `shape`.
    pub fn tensor(shape: [u32; DIM], usage: BufferUsages) -> Self {
        Self {
            shape,
            usage,
            label: None,
        }
    }

    /// The number of elements in this tensor.
    fn len(&self) -> u64 {
        self.shape.into_iter().map(|s| s as u64).product()
    }

    /// Sets the debug label of this tensor.
    pub fn label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Builds the gpu tensor.
    pub fn build<T: Pod>(self, device: &Device) -> GpuTensor<T, DIM> {
        let bytes_len = std::mem::size_of::<T>() as u64 * self.len();
        let buffer = device.create_buffer(&BufferDescriptor {
            label: self.label.as_deref(),
            size: bytes_len,
            usage: self.usage,
            mapped_at_creation: false,
        });

        GpuTensor {
            shape: self.shape,
            buffer,
            phantom: PhantomData,
        }
    }

    /// Builds the gpu tensor.
    pub fn build_uninit_encased<T: ShaderType>(self, device: &Device) -> GpuTensor<T, DIM> {
        let bytes_len = T::min_size().get() * self.len();
        let buffer = device.create_buffer(&BufferDescriptor {
            label: self.label.as_deref(),
            size: bytes_len,
            usage: self.usage,
            mapped_at_creation: false,
        });

        GpuTensor {
            shape: self.shape,
            buffer,
            phantom: PhantomData,
        }
    }

    /// Builds this tensor with raw bytes given for its initial value.
    pub fn build_bytes<T>(self, device: &Device, data: &[u8]) -> GpuTensor<T, DIM> {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: self.label.as_deref(),
            contents: bytemuck::cast_slice(data),
            usage: self.usage,
        });

        GpuTensor {
            shape: self.shape,
            buffer,
            phantom: PhantomData,
        }
    }

    /// Builds this tensor with raw bytes given for its initial value.
    pub fn build_encase<T>(self, device: &Device, data: impl AsRef<[T]>) -> GpuTensor<T, DIM>
    where
        T: ShaderType + ShaderSize + WriteInto,
    {
        let vector = data.as_ref();
        let mut bytes = vec![]; // TODO: can we avoid the allocation?
        let mut buffer = StorageBuffer::new(&mut bytes);
        buffer.write(vector).unwrap();
        self.build_bytes(device, &bytes)
    }
    /// Builds this tensor with an array of values given for its initial value.
    pub fn build_init<T: Pod>(self, device: &Device, data: &[T]) -> GpuTensor<T, DIM> {
        assert!(
            data.len() as u64 >= self.len(),
            "Incorrect number of elements provided for initializing Tensor.\
            Expected at least {}, found {}",
            self.len(),
            data.len()
        );

        let len = self.len();
        self.build_bytes::<T>(device, bytemuck::cast_slice(&data[..len as usize]))
    }
}

/// A tensor stored in the GPU.
///
/// When the tensor is a matrix, they are generally seen as being column-major.
pub struct GpuTensor<T, const DIM: usize> {
    shape: [u32; DIM],
    buffer: Buffer,
    phantom: PhantomData<T>,
}

impl<T, const DIM: usize> GpuTensor<T, DIM> {
    /// Does this tensor contain zero elements?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The number of elements in this tensor.
    pub fn len(&self) -> u64 {
        self.shape.into_iter().map(|s| s as u64).product()
    }

    /// The size, in bytes, of this tensor’s content.
    pub fn bytes_len(&self) -> u64
    where
        T: Pod,
    {
        std::mem::size_of::<T>() as u64 * self.len()
    }

    /// The size, in bytes, of this tensor’s content.
    pub fn bytes_len_encased(&self) -> u64
    where
        T: ShaderType,
    {
        T::min_size().get() * self.len()
    }

    /// Queues a buffer-to-buffer copy from `source` to `self`.
    ///
    /// Panics if the lengths do not match.
    pub fn copy_from(&self, encoder: &mut CommandEncoder, source: &GpuTensor<T, DIM>)
    where
        T: Pod,
    {
        assert_eq!(self.len(), source.len());
        encoder.copy_buffer_to_buffer(&source.buffer, 0, &self.buffer, 0, self.bytes_len())
    }

    pub fn copy_from_encased(&self, encoder: &mut CommandEncoder, source: &GpuTensor<T, DIM>)
    where
        T: ShaderType,
    {
        assert_eq!(self.len(), source.len());
        encoder.copy_buffer_to_buffer(&source.buffer, 0, &self.buffer, 0, self.bytes_len_encased())
    }

    /// Queues a buffer-to-buffer copy from `source` to `self`.
    pub fn copy_from_view<'a, Ordering>(
        &self,
        encoder: &mut CommandEncoder,
        source: impl Into<GpuTensorView<'a, T, Ordering, DIM>>,
    ) where
        T: Pod,
    {
        let source = source.into();
        assert_eq!(
            source.view_shape.size[0],
            if DIM == 0 { 1 } else { self.shape[0] }
        );

        encoder.copy_buffer_to_buffer(
            source.buffer,
            source.view_shape.offset as BufferAddress * size_of::<T>() as BufferAddress,
            &self.buffer,
            0,
            self.bytes_len(),
        )
    }

    /// The tensor’s shape (typically `[num_rows, num_cols, ...]`).
    pub fn shape(&self) -> [u32; DIM] {
        self.shape
    }

    /// The tensor’s underlying wgpu buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Extracts the underlying buffer.
    pub fn into_inner(self) -> Buffer {
        self.buffer
    }

    /// Builds a tensor view sharing the same shape, stride, and buffer, as `self`.
    pub fn as_view<Ordering: MatrixOrdering>(&self) -> GpuTensorView<'_, T, Ordering, DIM> {
        self.into()
    }

    // TODO: not sure if there is an official name for this operation.
    pub fn as_embedded_view<Ordering: MatrixOrdering, const DIM2: usize>(
        &self,
    ) -> GpuTensorView<'_, T, Ordering, DIM2> {
        assert!(
            DIM2 >= DIM,
            "Can only embed into a higher-order tensor view."
        );
        let mut embedded_shape = [1; DIM2];
        embedded_shape[..DIM].copy_from_slice(&self.shape[..DIM]);
        self.reshape(embedded_shape, None, None)
    }

    /// Reads the buffer’s content into a vector.
    pub async fn read_bytes<'a>(&'a self, device: &'a Device) -> anyhow::Result<BufferView<'a>> {
        // TODO: could probably be optimized?
        let buffer_slice = self.buffer.slice(..);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver) = async_channel::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                sender.send_blocking(v).unwrap()
            });
            let _ = device.poll(wgpu::PollType::wait());
            receiver.recv().await?.unwrap();
        }
        #[cfg(target_arch = "wasm32")]
        {
            let (sender, receiver) = async_channel::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = sender.force_send(v).unwrap();
            });
            device.poll(wgpu::PollType::wait())?;
            receiver.recv().await?.unwrap();
        }

        let data = buffer_slice.get_mapped_range();
        Ok(data)
    }

    /// Reads the buffer’s content into a slice.
    pub async fn read_to(&self, device: &Device, out: &mut [T]) -> anyhow::Result<()>
    where
        T: Pod,
    {
        let data = self.read_bytes(device).await?;
        let result = bytemuck::try_cast_slice(&data)?;
        out.copy_from_slice(result);
        drop(data);
        self.buffer.unmap();
        Ok(())
    }

    pub async fn slow_read(&self, gpu: &GpuInstance) -> Vec<T>
    where
        T: Pod,
    {
        // Create staging buffer and copy into it.
        let staging: GpuTensor<T, DIM> = TensorBuilder::tensor(
            self.shape(),
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        )
        .build(gpu.device());

        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        staging.copy_from(&mut encoder, self);
        gpu.queue().submit(Some(encoder.finish()));
        staging.read(gpu.device()).await.unwrap()
    }

    pub async fn slow_read_encased(&self, gpu: &GpuInstance) -> Vec<T>
    where
        T: ShaderType + ReadFrom + ShaderSize + CreateFrom,
    {
        // Create staging buffer and copy into it.
        let staging: GpuTensor<T, DIM> = TensorBuilder::tensor(
            self.shape(),
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        )
        .build_uninit_encased(gpu.device());

        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        staging.copy_from_encased(&mut encoder, self);
        gpu.queue().submit(Some(encoder.finish()));
        staging.read_encased(gpu.device()).await.unwrap()
    }

    /// Reads the buffer’s content into a vector.
    pub async fn read(&self, device: &Device) -> anyhow::Result<Vec<T>>
    where
        T: Pod,
    {
        let data = self.read_bytes(device).await?;
        let result = bytemuck::try_cast_slice(&data)?.to_vec();
        drop(data);
        self.buffer.unmap();
        Ok(result)
    }

    /// Reads the buffer’s content into a vector.
    pub async fn read_encased(&self, device: &Device) -> anyhow::Result<Vec<T>>
    where
        T: ShaderType + ReadFrom + ShaderSize + CreateFrom,
    {
        let data = self.read_bytes(device).await?;
        let mut result = vec![];
        let bytes = data.as_ref();
        let buffer = StorageBuffer::new(&bytes);
        buffer.read(&mut result)?;
        drop(data);
        self.buffer.unmap();
        Ok(result)
    }
}

// TODO: add a compile-time constraint for DIM1 <= DIM2
impl<'a, T, Ordering: MatrixOrdering, const DIM1: usize, const DIM2: usize>
    From<&'a GpuTensor<T, DIM1>> for GpuTensorView<'a, T, Ordering, DIM2>
{
    fn from(val: &'a GpuTensor<T, DIM1>) -> Self {
        val.as_embedded_view()
    }
}

/// A view over a tensor.
///
/// This is typically useful to extract a single matrix or column from a tensor. Note that,
/// currently, two elements from the same rows are required to be consecutive (row stride = 1).
#[derive(Copy, Clone)]
pub struct GpuTensorView<'a, T, Ordering, const DIM: usize> {
    view_shape: ViewShape,
    buffer: &'a Buffer,
    phantom: PhantomData<(T, Ordering)>,
}

impl<'a, T, Ordering, const DIM: usize> GpuTensorView<'a, T, Ordering, DIM> {
    /// The view’s shape.
    pub fn shape(&self) -> ViewShape {
        self.view_shape
    }

    /// The view’s underlying buffer.
    pub fn buffer(&self) -> &'a Buffer {
        self.buffer
    }
}

impl<T> GpuVectorView<'_, T> {
    /// Is this view empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of elements in this vector view.
    pub fn len(&self) -> u32 {
        self.view_shape.size[0]
    }

    pub fn rows(&self, i: u32, nrows: u32) -> Self {
        assert!(
            i + nrows <= self.len(),
            "Rows slice range out of bounds: {}..{}",
            i,
            i + nrows
        );
        GpuTensorView {
            view_shape: ViewShape {
                size: [nrows, 1, 1],
                stride: self.view_shape.stride,
                stride_mat: self.view_shape.stride_mat,
                offset: self.view_shape.offset + i,
            },
            buffer: self.buffer,
            phantom: PhantomData,
        }
    }
}

impl<'a, T, Ordering> GpuCubeView<'a, T, Ordering> {
    pub fn matrix(&self, matrix_id: u32) -> GpuMatrixView<'a, T, Ordering> {
        let [nrows, ncols, nmats] = self.view_shape.size;
        assert!(matrix_id < nmats);

        GpuTensorView {
            view_shape: ViewShape {
                size: [nrows, ncols, 1],
                stride: self.view_shape.stride,
                stride_mat: 1,
                offset: self.view_shape.offset + self.view_shape.stride_mat * matrix_id,
            },
            buffer: self.buffer,
            phantom: PhantomData,
        }
    }
}

impl<T, Ordering> GpuMatrixView<'_, T, Ordering> {
    pub fn columns(&self, first_col: u32, ncols: u32) -> Self {
        let nrows = self.view_shape.size[0];
        GpuTensorView {
            view_shape: ViewShape {
                size: [nrows, ncols, 1],
                stride: self.view_shape.stride,
                stride_mat: self.view_shape.stride_mat,
                offset: self.view_shape.offset + self.view_shape.stride * first_col,
            },
            buffer: self.buffer,
            phantom: PhantomData,
        }
    }

    pub fn rows(&self, first_row: u32, nrows: u32) -> Self {
        let ncols = self.view_shape.size[1];
        GpuTensorView {
            view_shape: ViewShape {
                size: [nrows, ncols, 1],
                stride: self.view_shape.stride,
                stride_mat: self.view_shape.stride_mat,
                offset: self.view_shape.offset + first_row,
            },
            buffer: self.buffer,
            phantom: PhantomData,
        }
    }
}

impl<T, const DIM: usize> GpuTensor<T, DIM> {
    pub fn reshape<Ordering: MatrixOrdering, const DIM2: usize>(
        &self,
        shape: [u32; DIM2],
        stride: Option<u32>,
        stride_mat: Option<u32>,
    ) -> GpuTensorView<'_, T, Ordering, DIM2> {
        assert!(shape.iter().product::<u32>() <= self.shape.iter().product::<u32>());

        let mut size = [1; 3];
        size[..DIM2].copy_from_slice(&shape[..DIM2]);

        let default_stride = if Ordering::is_column_major() {
            shape[0]
        } else {
            shape.get(1).copied().unwrap_or(1)
        };

        GpuTensorView {
            view_shape: ViewShape {
                size,
                stride: stride.unwrap_or(default_stride),
                stride_mat: stride_mat.unwrap_or(shape[0] * shape.get(1).copied().unwrap_or(1)),
                offset: 0,
            },
            buffer: &self.buffer,
            phantom: PhantomData,
        }
    }
}

impl<T> GpuMatrix<T> {
    /// Allocates a new matrix on the gpu with uninitialized elements.
    pub fn uninit(device: &Device, nrows: u32, ncols: u32, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        TensorBuilder::matrix(nrows, ncols, usage).build(device)
    }

    pub fn uninit_encased(device: &Device, nrows: u32, ncols: u32, usage: BufferUsages) -> Self
    where
        T: ShaderType,
    {
        TensorBuilder::matrix(nrows, ncols, usage).build_uninit_encased(device)
    }

    /// Allocates a new matrix on the gpu initialized from `matrix`.
    pub fn init<R: Dim, C: Dim, S: Storage<T, R, C> + IsContiguous>(
        device: &Device,
        matrix: &Matrix<T, R, C, S>,
        usage: BufferUsages,
    ) -> Self
    where
        T: Pod + nalgebra::Scalar,
    {
        TensorBuilder::matrix(matrix.nrows() as u32, matrix.ncols() as u32, usage)
            .build_init(device, matrix.as_slice())
    }

    /// Takes a view over the `i`-th column of `self`.
    pub fn column(&self, i: u32) -> GpuVectorView<'_, T> {
        GpuTensorView {
            view_shape: ViewShape {
                size: [self.shape[0], 1, 1],
                stride: 1,
                stride_mat: 1,
                offset: self.shape[0] * i,
            },
            buffer: &self.buffer,
            phantom: PhantomData,
        }
    }

    pub fn slice(&self, (i, j): (u32, u32), (nrows, ncols): (u32, u32)) -> GpuMatrixView<'_, T> {
        GpuTensorView {
            view_shape: ViewShape {
                size: [nrows, ncols, 1],
                stride: self.shape[0],
                stride_mat: self.shape[0] * self.shape[1],
                offset: i + j * nrows,
            },
            buffer: &self.buffer,
            phantom: PhantomData,
        }
    }

    pub fn columns(&self, first_col: u32, ncols: u32) -> GpuMatrixView<'_, T> {
        let nrows = self.shape[0];
        GpuTensorView {
            view_shape: ViewShape {
                size: [nrows, ncols, 1],
                stride: nrows,
                stride_mat: self.shape[0] * self.shape[1],
                offset: first_col * nrows,
            },
            buffer: &self.buffer,
            phantom: PhantomData,
        }
    }

    pub fn rows(&self, first_row: u32, nrows: u32) -> GpuMatrixView<'_, T> {
        let ncols = self.shape[1];
        GpuTensorView {
            view_shape: ViewShape {
                size: [nrows, ncols, 1],
                stride: self.shape[0],
                stride_mat: self.shape[0] * self.shape[1],
                offset: first_row,
            },
            buffer: &self.buffer,
            phantom: PhantomData,
        }
    }
}

impl<T> GpuVector<T> {
    /// Allocates a new vector on the gpu initialized from `vector`.
    ///
    /// If `T` implements `Pod`, use [`GpuMatrix::init`] instead.
    pub fn encase(device: &Device, vector: impl AsRef<[T]>, usage: BufferUsages) -> Self
    where
        T: ShaderType + ShaderSize + WriteInto,
    {
        let vector = vector.as_ref();
        TensorBuilder::vector(vector.len() as u32, usage).build_encase(device, vector)
    }

    /// Allocates a new uninitialized vector on the gpu for `len` elements of type `T`.
    pub fn uninit(device: &Device, len: u32, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        TensorBuilder::vector(len, usage).build(device)
    }

    /// Allocates a new uninitialized vector on the gpu for `len` elements of type `T`.
    pub fn uninit_encased(device: &Device, len: u32, usage: BufferUsages) -> Self
    where
        T: ShaderType,
    {
        TensorBuilder::vector(len, usage).build_uninit_encased(device)
    }

    /// Allocates a new vector on the gpu initialized from `vector`.
    ///
    /// If `T` does not implement `Pod`, use [`GpuMatrix::encase`] instead.
    pub fn init(device: &Device, vector: impl AsRef<[T]>, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        let v = vector.as_ref();
        TensorBuilder::vector(v.len() as u32, usage).build_init(device, v.as_ref())
    }

    /// Takes a view, over this vector, with `num_rows` rows starting at row `first_row`.
    pub fn rows(&self, first_row: u32, num_rows: u32) -> GpuVectorView<'_, T> {
        GpuTensorView {
            view_shape: ViewShape {
                size: [num_rows, 1, 1],
                stride: self.shape[0],
                stride_mat: self.shape[0],
                offset: first_row,
            },
            buffer: &self.buffer,
            phantom: PhantomData,
        }
    }
}

impl<T> GpuScalar<T> {
    /// Allocates a new gpu storage buffer with a single uninitialized element.
    pub fn uninit(device: &Device, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        TensorBuilder::scalar(usage).build(device)
    }

    pub fn uninit_encased(device: &Device, usage: BufferUsages) -> Self
    where
        T: ShaderType,
    {
        TensorBuilder::scalar(usage).build_uninit_encased(device)
    }

    /// Allocates a new gpu storage buffer with a single element initialized to `value`.
    pub fn init(device: &Device, value: T, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        TensorBuilder::scalar(usage).build_init(device, &[value])
    }
}
