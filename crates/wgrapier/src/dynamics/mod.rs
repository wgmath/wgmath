//! Rigid-body dynamics (forces, velocities, etc.)

pub use body::{
    BodyDesc, GpuBodySet, GpuForce, GpuLocalMassProperties, GpuVelocity, GpuWorldMassProperties,
    WgBody,
};
pub use constraint::{GpuTwoBodyConstraint, WgConstraint};
pub use integrate::WgIntegrate;
pub use sim_params::{GpuSimParams, WgSimParams};
pub use solver_jacobi::{JacobiSolverArgs, WgSolverJacobi};

pub mod body;
mod constraint;
pub mod integrate;
pub mod prefix_sum;
mod sim_params;
mod solver_jacobi;
