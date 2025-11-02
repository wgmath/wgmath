//! Rigid-body dynamics: forces, velocities, constraints, and solvers.
//!
//! This module contains all the components needed for simulating rigid-body dynamics on the GPU:
//!
//! # Core Components
//!
//! - **body**: Rigid body definitions including mass properties, velocities, and poses
//! - **constraint**: Contact constraints and their data structures for collision resolution
//! - **gravity**: Gravity force application
//! - **sim_params**: Global simulation parameters (timestep, solver iterations, etc.)
//!
//! # Constraint Solver
//!
//! - **solver_jacobi**: GPU-parallel constraint solver using graph coloring
//! - **coloring**: Graph coloring algorithms (TOPO-GC and Luby) for parallelizing constraint solving
//! - **warmstart**: Warmstarting mechanism to reuse impulses from previous frame
//!
//! # Utilities
//!
//! - **[`prefix_sum`]**: GPU prefix sum algorithm used for parallel data compaction
//!
//! # Physics Concepts
//!
//! ## Constraint-Based Dynamics
//!
//! The physics simulation uses a constraint-based approach where contacts between bodies
//! are modeled as constraints that restrict relative motion. Each constraint generates
//! impulses that modify body velocities to resolve penetrations and simulate friction.
//!
//! ## Parallel Solving with Graph Coloring
//!
//! To solve constraints in parallel on the GPU, a graph coloring algorithm assigns colors
//! to constraints such that no two constraints of the same color share a body. This allows
//! all constraints with the same color to be solved simultaneously without data races.
//!
//! ## TGS (Total Gauss-Seidel) Solver
//!
//! The default solver uses a variation of Gauss-Seidel iteration with substeps and
//! bias/no-bias phases for improved stability and convergence.

pub use body::{
    BodyDesc, GpuBodySet, GpuForce, GpuLocalMassProperties, GpuVelocity, GpuWorldMassProperties,
    WgBody,
};
pub use coloring::{ColoringArgs, WgColoring};
pub use constraint::{
    GpuTwoBodyConstraint, GpuTwoBodyConstraintBuilder, GpuTwoBodyConstraintInfos, WgConstraint,
};
pub use mprops_update::WgMpropsUpdate;
pub use sim_params::{GpuSimParams, WgSimParams};
pub use solver::{SolverArgs, WgSolver};
pub use warmstart::{WarmstartArgs, WgWarmstart};

pub mod body;
mod coloring;
mod constraint;
mod mprops_update;
pub mod prefix_sum;
mod sim_params;
mod solver;
mod warmstart;
