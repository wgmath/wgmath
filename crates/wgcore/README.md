# wgcore − utilities and abstractions for composable WGSL shaders

**wgcore** provides simple abstractions over shaders and gpu resources based on `wgpu`. It aims to:

- Expose thin wrappers that are as unsurprising as possible. We do not rely on complex compiler
  magic like bitcode generation in frameworks like `cust` and `rust-gpu`.
- Provide a proc-macro (through the `wgcore-derive` crate) to simplifies shader reuse across
  crates with very low boilerplate.
- No ownership of the gpu device and queue. While `wgcore` does expose a utility struct
  [`gpu::GpuInstance`] to initialize the compute unit, it is completely optional. All the features
  of `wgcore` remain usable if the gpu device and queue are already own by, e.g., a game engine.

## Shader composition

#### Basic usage

Currently, **wgcore** relies on [naga-oil](https://github.com/bevyengine/naga_oil) for shader
composition. Though we are keeping an eye on the ongoing [WESL](https://github.com/wgsl-tooling-wg)
effort for an alternative to `naga-oil`.

The main value added over `naga-oil` is the `wgcore::Shader` trait and proc-macro. This lets you
declare composable shaders very concisely. For example, if the WGSL sources are at the path
`./shader_sources.wgsl` relative to the `.rs` source file, all that’s needed for it to be composable
is to `derive` she `Shader` trait:

```rust ignore
#[derive(Shader)]
#[shader(src = "shader_source.wgsl")]
struct MyShader1;
```

Then it becomes immediately importable (assuming the `.wgsl` source itself contains a
`#define_import_path` statement) from another shader with the `shader(derive)` attribute:

```rust ignore
#[derive(Shader)]
#[shader(
    derive(MyShader1), // This shader depends on the `MyShader1` shader.
    src = "kernel.wgsl",  // Shader source code, will be embedded in the exe with `include_str!`.
)]
struct MyShader2;
```

Finally, if we want to use these shaders from another one which contains a kernel entry-point,
it is possible to declare `ComputePipeline` fields on the struct deriving `Shader`:

```rust ignore
#[derive(Shader)]
#[shader(
    derive(MyShader1, MyShader2),
    src = "kernel.wgsl",
)]
struct MyKernel {
    // Note that the field name has to match the kernel entry-point’s name.
    main: ComputePipeline,
}
```

This will automatically generate the necessary boiler-place for creating the compute pipeline
from a device: `MyKernel::from_device(device)`.

#### Some customization

The `Shader` proc-macro allows some customizations of the imported shaders:

- `src_fn = "function_name"`: allows the input sources to be modified by an arbitrary string
  transformation function before being compiled as a naga module. This enables any custom
  preprocessor to run before naga-oil.
- `shader_defs = "function_name"`: allows the declaration of shader definitions that can then be
  used in the shader in, e.g., `#ifdef MY_SHADER_DEF` statements (as well as `#if` statements and
  anything supported by the `naga-oil`’s shader definitions feature).
- `composable = false`: specifies that the shader does not exports any reusable symbols to other
  shaders. in particular, this **must** be specified if the shader sources doesn’t contain any
  `#define_import_path` statement.

```rust ignore
#[derive(Shader)]
#[shader(
    derive(MyShader1, MyShader2),
    src = "kernel.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs",
    composable = false
)]
struct MyKernel {
    main: ComputePipeline,
}