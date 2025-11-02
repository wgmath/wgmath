# wgebra − composable WGSL shaders for linear algebra

<p align="center">
  <img src="https://wgmath.rs/img/logo_wgebra.svg" alt="crates.io" height="200px">
</p>

----

The goal of **wgebra** is to especially be "[**nalgebra**](https://nalgebra.rs) on the gpu". It aims (but it isn’t there
yet) to expose linear algebra operations (including BLAS-like and LAPACK-like operations) as well as geometric types
(quaternions, similarities, etc.) as composable WGSL shaders and kernels.

## Reusable shader functions

**wgebra** exposes various reusable WGSL shaders to be composed with your owns. This exposes various functionalities
that are not covered by the mathematical functions included in the WebGPU standard:

- Low-dimensional matrix decompositions:
    - Inverse, Cholesky, LU, QR, Symmetric Eigendecomposition, for 2x2, 3x3, and 4x4 matrices.
    - Singular Values Decomposition for 2x2 and 3x3 matrices.
- Geometric transformations:
    - Quaternions (for 3D rotations).
    - Compact 2D rotation representation.
    - 2D and 3D similarities (rotations + translation + uniform scale).

## Kernels

**wgebra** exposes kernels for running common linear-algebra operations on vectors, matrices, and 3-tensors. In
particular:

- The product of two matrices: `Gemm` (including both `m1 * m2` and `transpose(m1) * m2`). Supports 3-tensors.
- The product of a matrix and a vector: `Gemv` (including both `m * v` and `transpose(m) * v`). Supports 3-tensors.
- Componentwise binary operations between two vectors (addition, subtraction, product, division, assignation).
- Reduction on a single vector (sum, product, min, max, squared norm).

## Using the library

To access the features of **wgebra** on your own Rust project, add the dependency to your `Cargo.toml`:

```toml
[dependencies]
wgebra = "0.2.0" # NOTE: set the version number to the latest.
```

Then shaders can be composed with your code, and kernels can be dispatched. For additional information, refer to
the [user-guide](https://wgmath.rs/docs/).

## Running tests

Tests can be run the same way as usual:

```sh
cargo test
```

## Benchmarks

There is currently no benchmark in the `wgebra` repository itself. However, some benchmarks of the matrix multiplication
kernels can be run from [wgml-bench](https://github.com/wgmath/wgml/tree/main/crates/wgml-bench).