//! GPU-accelerated rigid-body physics engine built on WebGPU/WGSL.
//!
//! **wgrapier** provides a high-performance physics simulation system that runs entirely on the GPU,
//! enabling massively parallel physics computation for thousands of rigid bodies. It is designed to
//! work seamlessly across platforms including web and desktop WebGPU.
//!
//! # See Also
//!
//! - [`wgparry`]: GPU collision detection library used by wgrapier.
//! - [`rapier`]: CPU-based physics engine that this crate is based on.
//! - [`wgcore`]: Foundation crate providing shader composition and GPU utilities.

#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;
#[cfg(feature = "dim2")]
pub extern crate wgparry2d as wgparry;
#[cfg(feature = "dim3")]
pub extern crate wgparry3d as wgparry;

pub mod dynamics;
pub mod pipeline;
