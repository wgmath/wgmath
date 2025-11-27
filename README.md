# wgmath − GPU scientific computing on every platform

<p align="center">
  <img src="https://wgmath.rs/img/wgmath_logo_w_padding.svg" alt="crates.io" height="200px">
</p>
<p align="center">
    <a href="https://discord.gg/vt9DJSW">
        <img src="https://img.shields.io/discord/507548572338880513.svg?logo=discord&colorB=7289DA">
    </a>
</p>

-----

**wgmath** is a set of [Rust](https://www.rust-lang.org/) libraries exposing
re-usable [WebGPU](https://www.w3.org/TR/WGSL/) shaders for scientific computing including:

- The [**wgcore** crate](https://github.com/wgmath/wgmath/tree/main/crates/wgcore), a centerpiece of the **wgmath**
  ecosystem, exposes a set of proc-macros to facilitate sharing and composing shaders across Rust libraries.
- Linear algebra with the [**wgebra** crate](https://github.com/wgmath/wgmath/tree/main/crates/wgebra).
- AI (Large Language Models) with the [**wgml** crate](https://github.com/wgmath/wgml/tree/main).
- Collision-detection with the
  [**wgparry2d** and **wgparry3d**](https://github.com/wgmath/wgmath/tree/main/crates/wgparry) crates.
- Rigid-body physics with the
  [**wgrapier2d** and **wgrapier3d**](https://github.com/wgmath/wgmath/tree/main/crates/wgrapier) crates.

By targeting WebGPU, these libraries run on most GPUs, including on mobile and on the web. It aims to promote open and
cross-platform GPU computing for scientific applications, a field currently strongly dominated by proprietary
solutions (like CUDA).

⚠️ All these libraries are still under heavy development and might be lacking some important features. Contributions
are welcome!

## Web Performance Notes

When running on the Web, best results are achieved with chrome-based browsers (including Edge).

Note that:

- We found WebGPU to be 10x slower on linux (tested on Ubuntu) compared to Windows or MacOS.
- We found Firefox (nightly) to be 10x slower than Chrome on all platforms (including MacOS/Windows).
- Keep in mind that some browsers don’t have WebGPU enabled by default and/or has an experimental
  WebGPU implementation (like safari) that might not work.

----

**See the readme of each individual crate (on the `crates` directory) for additional details.**

----
