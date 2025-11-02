## Unreleased

### Added

- Add `Shader::shader_module` to generate and return the shader’s `ShaderModule`.

### Changed

- Rename `Shader::set_absolute_path` to `Shader::set_wgsl_path`.
- Rename `Shader::absolute_path` to `Shader::wgsl_path`.
- Workgroup memory automatic zeroing is now **disabled** by default due to its significant
  performance impact.

## v0.2.2

### Fixed

- Fix crash in `HotReloadState` when targetting wasm.

## v0.2.1

### Fixed

- Fix build when targeting wasm.

## v0.2.0

### Added

- Add support for hot-reloading, see [#1](https://github.com/wgmath/wgmath/pull/1). This includes breaking changes to
  the `Shader` trait.
- Add support for shader overwriting, see [#1](https://github.com/wgmath/wgmath/pull/1).
