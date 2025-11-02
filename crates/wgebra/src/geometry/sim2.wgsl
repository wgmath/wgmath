#define_import_path wgebra::sim2
#import wgebra::rot2 as Rot


/// An 2D similarity representing a uniform scale, followed by a rotation, followed by a translation.
struct Sim2 {
    /// The similarity’s rotational part.
    rotation: Rot::Rot2,
    /// The similarity’s translational part.
    translation: vec2<f32>,
    /// The similarity’s scaling part.
    scale: f32,
}

/// The identity similarity.
fn identity() -> Sim2 {
    return Sim2(Rot::identity(), vec2(0.0f), 1.0f);
}

/// Multiplies two similarities.
fn mul(lhs: Sim2, rhs: Sim2) -> Sim2 {
    let rotation = Rot::mul(lhs.rotation, rhs.rotation);
    let translation = lhs.translation + Rot::mulVec(lhs.rotation, rhs.translation) * lhs.scale;
    return Sim2(rotation, translation, lhs.scale * rhs.scale);
}

/// Computes `inv(lhs) * rhs`.
fn invMul(lhs: Sim2, rhs: Sim2) -> Sim2 {
    // TODO: optimize?
    return mul(inv(lhs), rhs);
}

/// Inverts a similarity.
fn inv(sim: Sim2) -> Sim2 {
    let scale = 1.0f / sim.scale;
    let rotation = Rot::inv(sim.rotation);
    let translation = Rot::mulVec(rotation, -sim.translation) * scale;
    return Sim2(rotation, translation, scale);
}

/// Multiplies a similarity and a point (scales, rotates then translates the point).
fn mulPt(sim: Sim2, pt: vec2<f32>) -> vec2<f32> {
    return Rot::mulVec(sim.rotation, pt * sim.scale) + sim.translation;
}

/// Multiplies the inverse of a similarity and a point (inv-translates, inv-rotates, then inv-scales the point).
fn invMulPt(sim: Sim2, pt: vec2<f32>) -> vec2<f32> {
    return Rot::invMulVec(sim.rotation, (pt - sim.translation)) / sim.scale;
}

/// Multiplies a similarity and a unit vector (rotates the vector; the translation and scale are ignored).
fn mulUnitVec(sim: Sim2, vec: vec2<f32>) -> vec2<f32> {
    return Rot::mulVec(sim.rotation, vec);
}

/// Multiplies a similarity and a vector (scales and rotates the vector; the translation is ignored).
fn mulVec(sim: Sim2, vec: vec2<f32>) -> vec2<f32> {
    return Rot::mulVec(sim.rotation, vec) * sim.scale;
}

/// Multiplies the inverse of a similarity and a vector (inv-rotates then inv-scales the point; the translation is ignored).
fn invMulVec(sim: Sim2, vec: vec2<f32>) -> vec2<f32> {
    return Rot::invMulVec(sim.rotation, vec) / sim.scale;
}

/// Multiplies the inverse of a similarity and a unit vector.
///
/// This is similar to `invMulVec` but the scaling part of the similarity is ignored to preserve the vector’s unit size.
fn invMulUnitVec(sim: Sim2, vec: vec2<f32>) -> vec2<f32> {
    return Rot::invMulVec(sim.rotation, vec);
}
