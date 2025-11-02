//! GPU Radix Sort - Common Constants and Utilities
//!
//! This module defines shared constants for the GPU radix sort implementation.
//! The sort processes 32-bit keys in multiple passes, handling 4 bits per pass
//! using 16 histogram bins.
//!
//! Radix sort overview:
//! - Processes 32-bit keys in 8 passes (4 bits per pass)
//! - Each pass sorts into 16 bins (2^4)
//! - Stable sort (preserves relative order of equal keys)
//! - Sorts both keys and associated values
//!
//! Workgroup configuration:
//! - 256 threads per workgroup
//! - 4 elements per thread
//! - 1024 elements per workgroup (BLOCK_SIZE)
//!
//! Performance: O(k*n) where k=8 passes, linear in practice
//! Memory: O(n) for keys/values + O(workgroups * 16) for histograms
//!
//! Mostly copied from [brush](https://github.com/ArthurBrussee/brush)
//! (Apache-2.0 license).

#define_import_path wgparry::utils::sorting

const OFFSET: u32 = 42;
/// Workgroup size (threads per workgroup).
const WG: u32 = 256;

/// Number of bits processed per radix sort pass.
const BITS_PER_PASS: u32 = 4;
/// Number of histogram bins (2^BITS_PER_PASS).
const BIN_COUNT: u32 = 1u << BITS_PER_PASS;
/// Total histogram size across all threads in a workgroup.
const HISTOGRAM_SIZE: u32 = WG * BIN_COUNT;
/// Number of elements each thread processes.
const ELEMENTS_PER_THREAD: u32 = 4;

/// Total elements processed by one workgroup.
const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

/// Integer division with ceiling (rounds up).
fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}
