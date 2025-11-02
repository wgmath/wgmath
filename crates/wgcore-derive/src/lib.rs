//! Derive proc-macros for `wgcore`.

extern crate proc_macro;

use darling::util::PathList;
use darling::{FromDeriveInput, FromField};
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{Data, DataStruct, Path};

#[derive(FromDeriveInput, Clone)]
#[darling(attributes(shader))]
struct DeriveShadersArgs {
    #[darling(default)]
    pub derive: PathList,
    #[darling(default)]
    pub composable: Option<bool>,
    pub src: String,
    #[darling(default)]
    pub src_fn: Option<Path>,
    #[darling(default)]
    pub shader_defs: Option<Path>,
    #[darling(default = "default_crate_path")]
    pub krate: Path,
}

fn default_crate_path() -> Path {
    syn::parse_quote!(wgcore)
}

#[derive(FromField, Clone)]
#[darling(attributes(shader))]
struct DeriveShadersFieldArgs {
    #[darling(default)]
    pub kernel: Option<String>,
}

#[proc_macro_derive(Shader, attributes(shader))]
pub fn derive_shader(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::DeriveInput);
    let struct_identifier = &input.ident;

    let derive_shaders = match DeriveShadersArgs::from_derive_input(&input) {
        Ok(v) => v,
        Err(e) => {
            return e.write_errors().into();
        }
    };

    match &input.data {
        Data::Struct(DataStruct { fields, .. }) => {
            /*
             * Field attributes.
             */
            let mut kernels_to_build = vec![];
            let src_path = derive_shaders.src;

            for field in fields.iter() {
                let field_args = match DeriveShadersFieldArgs::from_field(field) {
                    Ok(v) => v,
                    Err(e) => {
                        return e.write_errors().into();
                    }
                };
                let ident = field.ident.as_ref().expect("unnamed fields not supported").into_token_stream();
                let kernel_name = field_args.kernel.map(|k| quote! { #k }).unwrap_or_else(|| quote! { stringify!(#ident) });

                let krate = &derive_shaders.krate;
                if fields.len() == 1 {
                    // Don't clone the module if there is only one field.
                    kernels_to_build.push(quote! {
                        #ident: #krate::utils::load_module(device, #kernel_name, module),
                    });
                } else {
                    kernels_to_build.push(quote! {
                        #ident: #krate::utils::load_module(device, #kernel_name, module.clone()),
                    });
                }
            }

            let krate = &derive_shaders.krate;
            let shader_defs = derive_shaders.shader_defs.map(|defs| quote! { #defs() })
                .unwrap_or_else(|| quote! { Default::default() });
            let raw_src = quote! {
                // First try to find a path from the shader registry.
                // If doesn't exist in the registry, try the absolute path.
                // If it doesn't exist in the absolute path, load the embedded string.
                if let Some(path) = Self::wgsl_path() {
                    // TODO: handle error
                    std::fs::read_to_string(path).unwrap()
                } else {
                    include_str!(#src_path).to_string()
                }
            };

            let src = derive_shaders.src_fn.map(|f| quote! { #f(&#raw_src) })
                .unwrap_or_else(|| quote! { #raw_src });
            let naga_module = quote! {
                Self::composer().and_then(|mut c|
                    c.make_naga_module(#krate::re_exports::naga_oil::compose::NagaModuleDescriptor {
                        source: &Self::src(),
                        file_path: Self::FILE_PATH,
                        shader_defs: #shader_defs,
                        ..Default::default()
                    })
                )
            };

            let from_device = if !kernels_to_build.is_empty() {
                quote! {
                    let module = #naga_module?;
                    Ok(Self {
                        #(
                            #kernels_to_build
                        )*
                    })
                }
            } else {
                quote! {
                    Ok(Self)
                }
            };

            /*
             * Derive shaders.
             */
            let to_derive: Vec<_> = derive_shaders
                .derive
                .iter()
                .map(|p| p.into_token_stream())
                .collect();
            let composable = derive_shaders.composable.unwrap_or(true);
            quote! {
                #[automatically_derived]
                impl #krate::shader::Shader for #struct_identifier {
                    const FILE_PATH: &'static str = #src_path;

                    fn from_device(device: &#krate::re_exports::Device) -> Result<Self, #krate::re_exports::ComposerError> {
                        #from_device
                    }

                    fn src() -> String {
                        #src
                    }

                    fn naga_module() -> Result<#krate::re_exports::wgpu::naga::Module, #krate::re_exports::ComposerError> {
                        #naga_module
                    }

                    fn wgsl_path() -> Option<std::path::PathBuf> {
                        if let Some(path) = #krate::ShaderRegistry::get().get_path::<#struct_identifier>() {
                            Some(path.clone())
                        } else {
                            // NOTE: this is a bit fragile, and won't work if the current working directory
                            //       isn't the root of the workspace the binary crate is being run from.
                            //       Ideally we need `proc_macro2::Span::source_file` but it is currently unstable.
                            //       See: https://users.rust-lang.org/t/how-to-get-the-macro-called-file-path-in-a-rust-procedural-macro/109613/5
                            std::path::Path::new(file!())
                                .parent()?
                                .join(Self::FILE_PATH)
                                .canonicalize().ok()
                        }
                    }

                    fn compose(composer: &mut #krate::re_exports::Composer) -> Result<(), #krate::re_exports::ComposerError> {
                        use #krate::composer::ComposerExt;
                        #(
                            #to_derive::compose(composer)?;
                        )*

                        if #composable {
                            composer
                                .add_composable_module_once(#krate::re_exports::ComposableModuleDescriptor {
                                    source: &Self::src(),
                                    file_path: Self::FILE_PATH,
                                    shader_defs: #shader_defs,
                                    ..Default::default()
                                })?;
                        }

                        Ok(())
                    }

                    /*
                     * Hot reloading.
                     */
                    fn watch_sources(state: &mut #krate::hot_reloading::HotReloadState) -> #krate::re_exports::notify::Result<()> {
                        #(
                            #to_derive::watch_sources(state)?;
                        )*

                        if let Some(path) = Self::wgsl_path() {
                            state.watch_file(&path)?;
                        }

                        Ok(())
                    }

                    fn needs_reload(state: &#krate::hot_reloading::HotReloadState) -> bool {
                        #(
                            if #to_derive::needs_reload(state) {
                                return true;
                            }
                        )*

                        Self::wgsl_path()
                            .map(|path| state.file_changed(&path))
                            .unwrap_or_default()
                    }
                }
            }
        }
        _ => unimplemented!(),
    }
        .into()
}
