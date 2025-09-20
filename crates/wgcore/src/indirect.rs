use crate::{self as wgcore, Shader};

#[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct DispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Shader)]
#[shader(src = "indirect.wgsl")]
pub struct WgIndirect;
