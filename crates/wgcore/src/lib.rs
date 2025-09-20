#![doc = include_str!("../README.md")]
// #![warn(missing_docs)]
#![allow(clippy::result_large_err)]

pub mod composer;
pub mod gpu;
pub mod hot_reloading;
pub mod indirect;
pub mod kernel;
pub mod shader;
pub mod shapes;
pub mod tensor;
pub mod timestamps;
pub mod utils;

pub use bytemuck::Pod;

pub use shader::{Shader, ShaderRegistry};
#[cfg(feature = "derive")]
pub use wgcore_derive::*;

/// Third-party modules re-exports.
pub mod re_exports {
    pub use bytemuck;
    pub use encase;
    pub use naga_oil::{
        self,
        compose::{ComposableModuleDescriptor, Composer, ComposerError, NagaModuleDescriptor},
    };
    pub use notify;
    pub use paste;
    pub use wgpu::{self, Device};
}

/// A macro that declares a test that will check compilation of the shader identified by the given
/// struct implementing `Shader`.
#[macro_export]
macro_rules! test_shader_compilation {
    ($ty: ident) => {
        wgcore::test_shader_compilation!($ty, wgcore);
    };

    ($ty: ident, $wgcore: ident) => {
        wgcore::test_shader_compilation!($ty, wgcore, Default::default());
    };

    ($ty: ident, $wgcore: ident, $shader_defs: expr) => {
        $wgcore::re_exports::paste::paste! {
            #[cfg(test)]
            mod [<test_shader_compiles_ $ty>] {
                use super::$ty;
                use naga_oil::compose::NagaModuleDescriptor;
                use $wgcore::Shader;
                use $wgcore::gpu::GpuInstance;
                use $wgcore::utils;

                #[futures_test::test]
                #[serial_test::serial]
                async fn shader_compiles() {
                    // Add a dumb entry point for testing.
                    let src = format!(
                        "{}
                            @compute @workgroup_size(1, 1, 1)
                            fn macro_generated_test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {{}}
                        ",
                        $ty::src()
                    );
                    let gpu = GpuInstance::new().await.unwrap();
                    let module = $ty::composer()
                        .unwrap()
                        .make_naga_module(NagaModuleDescriptor {
                            source: &src,
                            file_path: $ty::FILE_PATH,
                            shader_defs: $shader_defs,
                            ..Default::default()
                        })
                        .unwrap();
                    let _ = utils::load_module(gpu.device(), "macro_generated_test", module);
                }
            }
        }
    };
}
