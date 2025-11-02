# wgparry: cross-platform GPU collision-detection

<p align="center">
    <a href="https://crates.io/crates/wgparry3d">
        <img src="https://img.shields.io/crates/v/wgparry3d.svg?style=flat-square" alt="crates.io">
    </a>
</p>

**wgparry** is a GPU-accelerated collision-detection and geometry library for WebGPU. It aims to be "**parry** on the
GPU", providing geometric operations (collision-detection, ray-casting, point-projection, etc.) as composable WGSL
shaders and compute kernels.

⚠️ **This library is still under heavy development and is missing many features. Contributions are welcome!**

## Features

**wgparry** provides GPU implementations for:

- **Shapes**: Ball (sphere), cuboid, capsule, cone, cylinder, segment, triangle
- **Bounding volumes**: AABB (Axis-Aligned Bounding Box)
- **Broad-phase collision detection**: Brute-force and LBVH construction and traversal (Linear Bounding Volume
  Hierarchy)
- **Queries**: Ray-casting, point projection, contact computation, SAT (Separating Axis Test)
- **Contact manifold generation**: Polygonal features and contact points

## Usage

The library is available as two crates.

- `wgparry2d` for 2D collision detection
- `wgparry3d` for 3D collision detection

Add to your `Cargo.toml`:

```toml
[dependencies]
wgparry3d = "0.2"  # For 3D
# or
wgparry2d = "0.2"  # For 2D
```

## Resources

- [Main wgmath repository](https://github.com/wgmath/wgmath)
- [Documentation](https://wgmath.rs)
