#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

pub extern crate nalgebra as na;
#[cfg(feature = "dim2")]
pub extern crate parry2d as parry;
#[cfg(feature = "dim3")]
pub extern crate parry3d as parry;

use naga_oil::compose::ShaderDefValue;
use std::collections::HashMap;

/// Bounding volume data structures and GPU shaders for collision detection acceleration.
pub mod bounding_volumes;
/// Broad-phase collision detection algorithms implemented on the GPU.
///
/// Includes brute-force and LBVH (Linear Bounding Volume Hierarchy) implementations.
pub mod broad_phase;
/// Geometric query operations like ray-casting, point projection, and contact generation.
pub mod queries;
/// Geometric shape definitions and their GPU shader implementations.
pub mod shapes;
/// Utility functions and data structures, including GPU radix sort.
pub mod utils;

/// Returns shader definitions that depend on whether we are building the 2D or 3D version of this crate.
///
/// This function generates WGSL shader preprocessor definitions that allow shaders to adapt
/// based on dimensionality (2D vs 3D) and target platform (native vs WASM).
///
/// # Returns
///
/// A [`HashMap`] containing:
/// - `"DIM"`: Set to `2` for 2D builds (when `feature = "dim2"`) or `3` for 3D builds (when `feature = "dim3"`)
/// - `"NATIVE"`: Set to `1` for native targets or `0` for WASM targets
///
/// # Example
///
/// ```rust,ignore
/// let shader_defs = dim_shader_defs();
/// // On a 3D native build: {"DIM": 3, "NATIVE": 1}
/// // On a 2D WASM build: {"DIM": 2, "NATIVE": 0}
/// ```
pub fn dim_shader_defs() -> HashMap<String, ShaderDefValue> {
    let mut defs = HashMap::new();
    if cfg!(feature = "dim2") {
        defs.insert("DIM".to_string(), ShaderDefValue::UInt(2));
    } else {
        defs.insert("DIM".to_string(), ShaderDefValue::UInt(3));
    }

    if cfg!(target_arch = "wasm32") {
        defs.insert("NATIVE".to_string(), ShaderDefValue::UInt(0));
    } else {
        defs.insert("NATIVE".to_string(), ShaderDefValue::UInt(1));
    }

    defs
}

/// Substitutes type aliases in WGSL shader source code with their concrete dimension-specific types.
///
/// Since naga-oil doesn't support type aliases very well, this function performs textual
/// substitution to replace generic type names with their 2D or 3D equivalents before shader
/// compilation. This enables writing dimension-agnostic shader code that works for both
/// 2D and 3D builds.
///
/// # Parameters
///
/// - `src`: The WGSL shader source code containing generic type aliases
///
/// # Returns
///
/// The modified shader source with all aliases replaced by concrete types
///
/// # Substituted Aliases
///
/// For 2D builds (feature = "dim2"):
/// - `Transform` → `Pose::Sim2` (2D similarity transformation)
/// - `AngVector` → `f32` (scalar angle)
/// - `Vector` → `vec2<f32>` (2D vector)
///
/// For 3D builds (feature = "dim3"):
/// - `Transform` → `Pose::Sim3` (3D similarity transformation)
/// - `AngVector` → `vec3<f32>` (3D angular vector)
/// - `Vector` → `vec3<f32>` (3D vector)
///
/// # Example
///
/// ```rust,ignore
/// let shader = "fn distance(a: Vector, b: Vector) -> f32 { ... }";
/// let result = substitute_aliases(shader);
/// // In 3D: "fn distance(a: vec3<f32>, b: vec3<f32>) -> f32 { ... }"
/// ```
pub fn substitute_aliases(src: &str) -> String {
    #[cfg(feature = "dim2")]
    return src
        .replace("Transform", "Pose::Sim2")
        .replace("AngVector(", "f32(")
        .replace("AngVector", "f32")
        .replace("Vector(", "vec2<f32>(")
        .replace("Vector", "vec2<f32>");
    #[cfg(feature = "dim3")]
    return src
        .replace("Transform", "Pose::Sim3")
        .replace("AngVector(", "vec3(")
        .replace("AngVector", "vec3<f32>")
        .replace("Vector(", "vec3<f32>(")
        .replace("Vector", "vec3<f32>");
}

// NOTE: the modules below were copied from parry. Should we just add a dependency to parry?

/// Compilation flags dependent aliases for mathematical types.
#[cfg(feature = "dim3")]
pub mod math {
    use na::{
        Isometry3, Matrix3, Point3, Translation3, UnitQuaternion, UnitVector3, Vector3, Vector6,
        U3, U6,
    };
    use wgebra::GpuSim3;

    /// The default tolerance used for geometric operations.
    pub const DEFAULT_EPSILON: f32 = f32::EPSILON;

    /// The dimension of the space.
    pub const DIM: usize = 3;

    /// The dimension of the space multiplied by two.
    pub const TWO_DIM: usize = DIM * 2;

    /// The dimension of the ambient space.
    pub type Dim = U3;

    /// The dimension of a spatial vector.
    pub type SpatialDim = U6;

    /// The dimension of the rotations.
    pub type AngDim = U3;

    /// The point type.
    pub type Point<N> = Point3<N>;

    /// The angular vector type.
    pub type AngVector<N> = Vector3<N>;

    /// The vector type.
    pub type Vector<N> = Vector3<N>;

    /// The unit vector type.
    pub type UnitVector<N> = UnitVector3<N>;

    /// The matrix type.
    pub type Matrix<N> = Matrix3<N>;

    /// The vector type with dimension `SpatialDim × 1`.
    pub type SpatialVector<N> = Vector6<N>;

    /// The orientation type.
    pub type Orientation<N> = Vector3<N>;

    /// The transformation matrix type.
    pub type Isometry<N> = Isometry3<N>;

    /// The rotation matrix type.
    pub type Rotation<N> = UnitQuaternion<N>;

    /// The translation type.
    pub type Translation<N> = Translation3<N>;

    /// The angular inertia of a rigid body.
    pub type AngularInertia<N> = Matrix3<N>;

    /// The principal angular inertia of a rigid body.
    pub type PrincipalAngularInertia<N> = Vector3<N>;

    /// A matrix that represent the cross product with a given vector.
    pub type CrossMatrix<N> = Matrix3<N>;

    /// A vector with a dimension equal to the maximum number of degrees of freedom of a rigid body.
    pub type SpacialVector<N> = Vector6<N>;

    // /// A 3D symmetric-definite-positive matrix.
    // pub type SdpMatrix<N> = crate::utils::SdpMatrix3<N>;

    /// A 3D similarity with layout compatible with the corresponding wgsl struct.
    pub type GpuSim = GpuSim3;
}

/// Compilation flags dependent aliases for mathematical types.
#[cfg(feature = "dim2")]
pub mod math {
    use na::{
        Isometry2, Matrix2, Point2, Translation2, UnitComplex, UnitVector2, Vector1, Vector2,
        Vector3, U1, U2,
    };
    use wgebra::GpuSim2;

    /// The default tolerance used for geometric operations.
    pub const DEFAULT_EPSILON: f32 = f32::EPSILON;

    /// The dimension of the space.
    pub const DIM: usize = 2;

    /// The dimension of the space multiplied by two.
    pub const TWO_DIM: usize = DIM * 2;

    /// The dimension of the ambient space.
    pub type Dim = U2;

    /// The dimension of the rotations.
    pub type AngDim = U1;

    /// The point type.
    pub type Point<N> = Point2<N>;

    /// The angular vector type.
    pub type AngVector<N> = N;

    /// The vector type.
    pub type Vector<N> = Vector2<N>;

    /// The unit vector type.
    pub type UnitVector<N> = UnitVector2<N>;

    /// The matrix type.
    pub type Matrix<N> = Matrix2<N>;

    /// The orientation type.
    pub type Orientation<N> = Vector1<N>;

    /// The transformation matrix type.
    pub type Isometry<N> = Isometry2<N>;

    /// The rotation matrix type.
    pub type Rotation<N> = UnitComplex<N>;

    /// The translation type.
    pub type Translation<N> = Translation2<N>;

    /// The angular inertia of a rigid body.
    pub type AngularInertia<N> = N;

    /// The principal angular inertia of a rigid body.
    pub type PrincipalAngularInertia<N> = N;

    /// A matrix that represent the cross product with a given vector.
    pub type CrossMatrix<N> = Vector2<N>;

    /// A vector with a dimension equal to the maximum number of degrees of freedom of a rigid body.
    pub type SpacialVector<N> = Vector3<N>;

    // /// A 2D symmetric-definite-positive matrix.
    // pub type SdpMatrix<N> = crate::utils::SdpMatrix2<N>;

    /// A 2D similarity with layout compatible with the corresponding wgsl struct.
    pub type GpuSim = GpuSim2;
}
