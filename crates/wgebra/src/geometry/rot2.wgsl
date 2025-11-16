#define_import_path wgebra::rot2

#import wgebra::trig as Trig


/// Compact representation of a 2D rotation.
struct Rot2 {
    cos_sin: vec2<f32>
}

/// Returns `true` if `rot` isn’t zero.
///
/// Failible functions that return a Rot2 will generally return zero
/// as the value to indicate failure.
fn is_valid(rot: Rot2) -> bool {
    return rot.cos_sin.x != 0.0 || rot.cos_sin.y != 0.0;
}

/// Initializes a 2D rotation from an angle (radians).
fn fromAngle(angle: f32) -> Rot2 {
    return Rot2(vec2(cos(angle), sin(angle)));
}

/// Computes the rotation `R` required such that the `y` component of `R * v` is zero.
///
/// Returns `Rot2()` (i.e. Rot2 filled with zeros) if no rotation is needed (i.e. if `v.y == 0`). Otherwise, this returns
/// the rotation `R` such that `R * v = [ |v|, 0.0 ]^t` where `|v|` is the norm of `v`.
fn cancel_y(v: vec2<f32>) -> Rot2 {
    if v.y != 0.0 {
        let r = sign(v.x) / length(v);
        let cos_sin = vec2(v.x, -v.y) * r;
        return Rot2(cos_sin);
    } else {
        return Rot2();
    }
}

/// The quaternion representing an identity rotation.
fn identity() -> Rot2 {
    return Rot2(vec2(1.0, 0.0));
}

fn toMatrix(r: Rot2) -> mat2x2<f32> {
    return mat2x2(
        vec2(r.cos_sin.x, r.cos_sin.y),
        vec2(-r.cos_sin.y, r.cos_sin.x)
    );
}

/// The rotation angle (in radians).
fn angle(r: Rot2) -> f32 {
    return Trig::stable_atan2(r.cos_sin.y, r.cos_sin.x);
}

/// The inverse of a 2d rotation.
fn inv(r: Rot2) -> Rot2 {
    return Rot2(vec2(r.cos_sin.x, -r.cos_sin.y));
}

/// Multiplication of two 2D rotations.
fn mul(lhs: Rot2, rhs: Rot2) -> Rot2 {
    let new_cos = lhs.cos_sin.x * rhs.cos_sin.x - lhs.cos_sin.y * rhs.cos_sin.y;
    let new_sin = lhs.cos_sin.y * rhs.cos_sin.x + lhs.cos_sin.x * rhs.cos_sin.y;
    return Rot2(vec2(new_cos, new_sin));
}

/// Multiplies a 2D rotation by a vector (rotates the vector).
fn mulVec(r: Rot2, v: vec2<f32>) -> vec2<f32> {
    return vec2(r.cos_sin.x * v.x - r.cos_sin.y * v.y, r.cos_sin.y * v.x + r.cos_sin.x * v.y);
}

/// Multiplies the inverse of a 2D rotation by a vector (applies inverse rotation to the vector).
fn invMulVec(r: Rot2, v: vec2<f32>) -> vec2<f32> {
    return vec2(r.cos_sin.x * v.x + r.cos_sin.y * v.y, -r.cos_sin.y * v.x + r.cos_sin.x * v.y);
}

// Apply the rotation to rows i and i + 1 to the given 3x3 matrix.
fn rotate_rows3(rot: Rot2, m: ptr<function, mat3x3<f32>>, i: u32) {
    for (var r = 0; r < 3; r++) {
        let v = vec2((*m)[i][r], (*m)[i + 1][r]);
        let rv = invMulVec(rot, v);
        (*m)[i][r] = rv.x;
        (*m)[i + 1][r] = rv.y;
    }
}

// Apply the rotation to rows i and i + 1 to the given 4x4 matrix.
fn rotate_rows4(rot: Rot2, m: ptr<function, mat4x4<f32>>, i: u32) {
    for (var r = 0; r < 4; r++) {
        let v = vec2((*m)[i][r], (*m)[i + 1][r]);
        let rv = invMulVec(rot, v);
        (*m)[i][r] = rv.x;
        (*m)[i + 1][r] = rv.y;
    }
}