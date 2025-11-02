#define_import_path wgebra::sim3
#import wgebra::quat as Rot


/// An 3D similarity representing a uniform scale, followed by a rotation, followed by a translation.
struct Sim3 {
    /// The similarity’s rotational part.
    rotation: Rot::Quat,
    /// The similarity’s translational (xyz) and scaling (w) part.
    translation_scale: vec4<f32>
}

/// The identity similarity.
fn identity() -> Sim3 {
    return Sim3(Rot::identity(), vec4(0.0f, 0.0f, 0.0f, 1.0f));
}

/// Multiplies two similarities.
fn mul(lhs: Sim3, rhs: Sim3) -> Sim3 {
    let rotation = Rot::mul(lhs.rotation, rhs.rotation);
    let translation = lhs.translation_scale.xyz + Rot::mulVec(lhs.rotation, rhs.translation_scale.xyz) * lhs.translation_scale.w;
    return Sim3(rotation, vec4(translation, lhs.translation_scale.w * rhs.translation_scale.w));
}

/// Inverts a similarity.
fn inv(sim: Sim3) -> Sim3 {
    let scale = 1.0f / sim.translation_scale.w;
    let rotation = Rot::inv(sim.rotation);
    let translation = Rot::mulVec(rotation, -sim.translation_scale.xyz) * scale;
    return Sim3(rotation, vec4(translation, scale));
}

/// Computes `inv(lhs) * rhs`.
fn invMul(lhs: Sim3, rhs: Sim3) -> Sim3 {
    // TODO: optimize?
    return mul(inv(lhs), rhs);
}

/// Multiplies a similarity and a point (scales, rotates then translates the point).
fn mulPt(sim: Sim3, pt: vec3<f32>) -> vec3<f32> {
    return Rot::mulVec(sim.rotation, pt * sim.translation_scale.w) + sim.translation_scale.xyz;
}

/// Multiplies the inverse of a similarity and a point (inv-translates, inv-rotates, then inv-scales the point).
fn invMulPt(sim: Sim3, pt: vec3<f32>) -> vec3<f32> {
    return Rot::invMulVec(sim.rotation, (pt - sim.translation_scale.xyz)) / sim.translation_scale.w;
}

/// Multiplies a similarity and a unit vector (rotates the vector; the scale and translation are ignored).
fn mulUnitVec(sim: Sim3, vec: vec3<f32>) -> vec3<f32> {
    return Rot::mulVec(sim.rotation, vec);
}

/// Multiplies a similarity and a vector (scales and rotates the vector; the translation is ignored).
fn mulVec(sim: Sim3, vec: vec3<f32>) -> vec3<f32> {
    return Rot::mulVec(sim.rotation, vec) * sim.translation_scale.w;
}

/// Multiplies the inverse of a similarity and a vector (inv-rotates then inv-scales the point; the translation is ignored).
fn invMulVec(sim: Sim3, vec: vec3<f32>) -> vec3<f32> {
    return Rot::invMulVec(sim.rotation, vec) / sim.translation_scale.w;
}

/// Multiplies the inverse of a similarity and a unit vector.
///
/// This is similar to `invMulVec` but the scaling part of the similarity is ignored to preserve the vector’s unit size.
fn invMulUnitVec(sim: Sim3, vec: vec3<f32>) -> vec3<f32> {
    return Rot::invMulVec(sim.rotation, vec);
}
