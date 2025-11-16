#define_import_path wgebra::quat

/// A unit quaternion representing a rotation.
struct Quat {
    /// The quaternion’s coordinates (i, j, k, w).
    coords: vec4<f32>,
}

/// The quaternion representing an identity rotation.
fn identity() -> Quat {
    return Quat(vec4(0.0, 0.0, 0.0, 1.0));
}

/// Convert an axis-angle (represented as the axis multiplied by the angle) to
/// a quaternion.
fn fromScaledAxis(axisangle: vec3<f32>) -> Quat {
    let angle = length(axisangle);
    let is_zero = f32(angle == 0.0);

    if angle == 0.0 {
        return identity();
    } else {
        let hs = sin(angle / 2.0);
        let hc = cos(angle / 2.0);
        let axis = axisangle / angle;
        return Quat(vec4(axis * hs, hc));
    }
}

// Converts this quaternion to a rotation matrix.
fn toMatrix(quat: Quat) -> mat3x3<f32> {
    let i = quat.coords.x;
    let j = quat.coords.y;
    let k = quat.coords.z;
    let w = quat.coords.w;

    let ww = w * w;
    let ii = i * i;
    let jj = j * j;
    let kk = k * k;
    let ij = i * j * 2.0;
    let wk = w * k * 2.0;
    let wj = w * j * 2.0;
    let ik = i * k * 2.0;
    let jk = j * k * 2.0;
    let wi = w * i * 2.0;

    return mat3x3(
        vec3(ww + ii - jj - kk, wk + ij, ik - wj),
        vec3(ij - wk, ww - ii + jj - kk, wi + jk),
        vec3(wj + ik, jk - wi, ww - ii - jj + kk),
    );
}

/// Normalizes this quaternion again using a first-order Taylor approximation.
/// This is useful when repeated computations might cause a drift in the norm
/// because of float inaccuracies.
fn renormalizeFast(q: Quat) -> Quat {
    let sq_norm = dot(q.coords, q.coords);
    return Quat(q.coords * (0.5 * (3.0 - sq_norm)));
}

fn imag(q: Quat) -> vec3<f32> {
    return q.coords.xyz;
}

/// The inverse (conjugate) of a unit quaternion.
fn inv(q: Quat) -> Quat {
    return Quat(vec4(-q.coords.xyz, q.coords.w));
}

/// Multiplies two quaternions (combines their rotations).
fn mul(lhs: Quat, rhs: Quat) -> Quat {
    let scalar = lhs.coords.w * rhs.coords.w - dot(lhs.coords.xyz, rhs.coords.xyz);
    let v = cross(lhs.coords.xyz, rhs.coords.xyz) + lhs.coords.w * rhs.coords.xyz + rhs.coords.w * lhs.coords.xyz;
    return Quat(vec4(v, scalar));
}

/// Multiplies a quaternion by a vector (rotates the vector).
fn mulVec(q: Quat, v: vec3<f32>) -> vec3<f32> {
    let t = cross(q.coords.xyz, v) * 2.0;
    let c = cross(q.coords.xyz, t);
    return t * q.coords.w + c + v;
}

/// Multiplies a quaternion’s inverse by a vector (inverse-rotates the vector).
fn invMulVec(q: Quat, v: vec3<f32>) -> vec3<f32> {
    let t = cross(q.coords.xyz, v) * 2.0;
    let c = cross(q.coords.xyz, t);
    return t * -q.coords.w + c + v;
}

fn diff_conj1_2(a: Quat, b: Quat) -> mat3x3<f32> {
        let v1 = imag(a);
        let v2 = imag(b);
        let w1 = a.coords.w;
        let w2 = b.coords.w;

        // TODO: this can probably be optimized a lot by unrolling the ops.
        return (tensor_prod(v1, v2) + diag(w1 * w2)
            - cross_matrix(v1 * w2 + v2 * w1)
            + cross_matrix(v1) * cross_matrix(v2))
            * 0.5;
}

/*
 * Misc utilities.
 */
fn tensor_prod(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    // NOTE: keep in mind that the matrix is initialized column
    //       by column. So the element order look transposed compared
    //       to traditional mathematical notation.
    return mat3x3(
        vec3(a.x * b.x, a.y * b.x, a.z * b.x),
        vec3(a.x * b.y, a.y * b.y, a.z * b.y),
        vec3(a.x * b.z, a.y * b.z, a.z * b.z),
    );
}

fn diag(elt: f32) -> mat3x3<f32> {
    return mat3x3(
        vec3(elt, 0.0, 0.0),
        vec3(0.0, elt, 0.0),
        vec3(0.0, 0.0, elt),
    );
}

fn cross_matrix(r: vec3<f32>) -> mat3x3<f32> {
    // NOTE: keep in mind that the matrix is initialized column
    //       by column. So the element order look transposed compared
    //       to traditional mathematical notation.
    return mat3x3(
        vec3(0.0, r.z, -r.y),
        vec3(-r.z, 0.0, r.x),
        vec3(r.y, -r.x, 0.0),
    );
}