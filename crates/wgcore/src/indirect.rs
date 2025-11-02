#[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct DispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[cfg(feature = "derive")]
use crate::Shader;

#[cfg(feature = "derive")]
#[derive(Shader)]
#[shader(src = "indirect.wgsl", krate = "crate")]
pub struct WgIndirect;
