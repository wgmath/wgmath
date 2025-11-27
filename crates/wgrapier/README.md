# wgrapier: cross-platform GPU physics simulation

<p align="center">
    <a href="https://crates.io/crates/wgrapier3d">
        <img src="https://img.shields.io/crates/v/wgrapier3d.svg?style=flat-square" alt="crates.io">
    </a>
</p>

**wgrapier** is a GPU-accelerated rigid-body physics engine for WebGPU. It aims to be "**rapier** on the GPU", providing
high-performance physics simulation. By targeting WebGPU, **wgrapier** runs on most GPUs, including web browsers.

⚠️ **This library is still under heavy development and is missing many features. Contributions are welcome!**

## Features

**wgrapier** provides GPU (WGSL) implementations for:

- **Rigid-body dynamics**: Integration, gravity, constraint solving
- **Collision detection**: Built on top of [wgparry](../wgparry) for broad-phase and narrow-phase detection
- **Constraint solver**: Parallel constraint resolution with graph coloring (Jacobi and TGS-Soft)
- **Warmstarting**: Improved convergence using cached constraint impulses

The physics pipeline runs entirely on the GPU.

## Usage

The library is available as two crates depending on your needs:

- `wgrapier2d` for 2D physics simulation
- `wgrapier3d` for 3D physics simulation

Add to your `Cargo.toml`:

```toml
[dependencies]
wgrapier3d = "0.2"  # For 3D
# or
wgrapier2d = "0.2"  # For 2D
```

## Examples

Example programs demonstrating various physics scenarios can be found in the `crates/examples2d` and `crates/examples3d`
directories. Run examples with:

```bash
# Run natively
cargo run --release --bin all_examples2
cargo run --release --bin all_examples3

# Run on the browser
rustup target add wasm32-unknown-unknown # Run this only once.
cargo install wasm-server-runner         # Run this only once.
cargo run --release --bin all_examples2 --target wasm32-unknown-unknown
cargo run --release --bin all_examples3 --target wasm32-unknown-unknown
```

## Web Performance Notes

When running on the Web, best results are achieved with chrome-based browsers (including Edge).

Note that:

- We found WebGPU to be 10x slower on linux (tested on Ubuntu) compared to Windows or MacOS.
- We found Firefox (nightly) to be 10x slower than Chrome on all platforms (including MacOS/Windows).
- Keep in mind that some browsers don’t have WebGPU enabled by default and/or has an experimental
  WebGPU implementation (like safari) that might not work.

## Resources

- [wgmath repository](https://github.com/wgmath/wgmath)
- [Documentation](https://wgmath.rs)
