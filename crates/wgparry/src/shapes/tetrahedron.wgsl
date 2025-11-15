//! Tetrahedron Shape Module
//!
//! This module provides the tetrahedron shape definition from its four vertices.

#define_import_path wgparry::tetrahedron

#import wgparry::projection as Proj

// TODO: group all the epsilon in the same place.
const FLT_EPS: f32 = 1.0e-7;

/// A tetrahedron defined by four vertices.
struct Tetrahedron {
    /// First vertex of the tetrahedron.
    a: Vector,
    /// Second vertex of the tetrahedron.
    b: Vector,
    /// Third vertex of the tetrahedron.
    c: Vector,
    /// Fourth vertex of the tetrahedron.
    d: Vector,
}

struct EdgeCheck {
    proj: Proj::ProjectionWithLocation,
    du: f32,
    dv: f32,
    valid: bool,
}

/*
 * Voronoï regions of edges.
 */
fn check_edge(
    i: u32,
    a: Vector,
    b: Vector,
    nabc: Vector,
    nabd: Vector,
    ap: Vector,
    ab: Vector,
    ap_ab: f32, /*ap_ac: f32, ap_ad: f32,*/
    bp_ab: f32, /*bp_ac: f32, bp_ad: f32*/
) -> EdgeCheck {
    let ab_ab = ap_ab - bp_ab;

    // NOTE: The following avoids the subsequent cross and dot products but are not
    // numerically stable.
    //
    // let dabc  = ap_ab * (ap_ac - bp_ac) - ap_ac * ab_ab;
    // let dabd  = ap_ab * (ap_ad - bp_ad) - ap_ad * ab_ab;

    let ap_x_ab = cross(ap, ab);
    let dabc = dot(ap_x_ab, nabc);
    let dabd = dot(ap_x_ab, nabd);

    // TODO: the case where ab_ab == 0.0 is not well defined.
    if ab_ab != 0.0 && dabc >= 0.0 && dabd >= 0.0 && ap_ab >= 0.0 && ap_ab <= ab_ab {
        // Voronoi region of `ab`.
        let u = ap_ab / ab_ab;
        let bcoords = vec2(1.0 - u, u);
        let res = a + ab * u;
        let proj = Proj::edge(res, bcoords, i, false);
        return EdgeCheck(proj, dabc, dabd, true);
    } else {
        return EdgeCheck(Proj::ProjectionWithLocation(), dabc, dabd, false);
    }
}

/*
 * Voronoï regions of faces.
 */
struct FaceCheck {
    proj: Proj::ProjectionWithLocation,
    valid: bool,
}

fn check_face(
    i: u32,
    a: Vector,
    b: Vector,
    c: Vector,
    ap: Vector,
    bp: Vector,
    cp: Vector,
    ab: Vector,
    ac: Vector,
    ad: Vector,
    dabc: f32,
    dbca: f32,
    dacb: f32,
    /* ap_ab: f32, bp_ab: f32, cp_ab: f32,
    ap_ac: f32, bp_ac: f32, cp_ac: f32, */
) -> FaceCheck {
    if dabc < 0.0 && dbca < 0.0 && dacb < 0.0 {
        let n = cross(ab, ac); // TODO: is is possible to avoid this cross product?
        if dot(n, ad) * dot(n, ap) < 0.0 {
            // Voronoï region of the face.

            // NOTE:
            // The following avoids expansive computations but are not very
            // numerically stable.
            //
            // let va = bp_ab * cp_ac - cp_ab * bp_ac;
            // let vb = cp_ab * ap_ac - ap_ab * cp_ac;
            // let vc = ap_ab * bp_ac - bp_ab * ap_ac;

            // NOTE: the normalization may fail even if the dot products
            // above were < 0. This happens, e.g., when we use fixed-point
            // numbers and there are not enough decimal bits to perform
            // the normalization.
            let n_length = length(n);

            if n_length < FLT_EPS {
                return FaceCheck(Proj::ProjectionWithLocation(), false);
            }

            let normal = n / n_length;
            let vc = dot(normal, cross(ap, bp));
            let va = dot(normal, cross(bp, cp));
            let vb = dot(normal, cross(cp, ap));

            let denom = va + vb + vc;
            let inv_denom = 1.0 / denom;

            let bcoords = vec3(va * inv_denom, vb * inv_denom, vc * inv_denom);
            let res = a * bcoords[0] + b * bcoords[1] + c * bcoords[2];
            return FaceCheck(Proj::face(res, bcoords, i, false), true);
        }
    }

    return FaceCheck(Proj::ProjectionWithLocation(), false);
}

fn project_local_point_and_get_location(shape: Tetrahedron, pt: Vector, solid: bool) -> Proj::ProjectionWithLocation {
    let ab = shape.b - shape.a;
    let ac = shape.c - shape.a;
    let ad = shape.d - shape.a;
    let ap = pt - shape.a;

    /*
     * Voronoï regions of vertices.
     */
    let ap_ab = dot(ap, ab);
    let ap_ac = dot(ap, ac);
    let ap_ad = dot(ap, ad);

    if ap_ab <= 0.0 && ap_ac <= 0.0 && ap_ad <= 0.0 {
        // Voronoï region of `a`.
        return Proj::vertex(shape.a, 0, false);
    }

    let bc = shape.c - shape.b;
    let bd = shape.d - shape.b;
    let bp = pt - shape.b;

    let bp_bc = dot(bp, bc);
    let bp_bd = dot(bp, bd);
    let bp_ab = dot(bp, ab);

    if bp_bc <= 0.0 && bp_bd <= 0.0 && bp_ab >= 0.0 {
        // Voronoï region of `b`.
        return Proj::vertex(shape.b, 1, false);
    }

    let cd = shape.d - shape.c;
    let cp = pt - shape.c;

    let cp_ac = dot(cp, ac);
    let cp_bc = dot(cp, bc);
    let cp_cd = dot(cp, cd);

    if cp_cd <= 0.0 && cp_bc >= 0.0 && cp_ac >= 0.0 {
        // Voronoï region of `c`.
        return Proj::vertex(shape.c, 2, false);
    }

    let dp = pt - shape.d;

    let dp_cd = dot(dp, cd);
    let dp_bd = dot(dp, bd);
    let dp_ad = dot(dp, ad);

    if dp_ad >= 0.0 && dp_bd >= 0.0 && dp_cd >= 0.0 {
        // Voronoï region of `d`.
        return Proj::vertex(shape.d, 3, false);
    }

    // Voronoï region of ab.
    //            let bp_ad = bp_bd + bp_ab;
    //            let bp_ac = bp_bc + bp_ab;
    let nabc = cross(ab, ac);
    let nabd = cross(ab, ad);
    let res_ab = check_edge(
        0, shape.a, shape.b, nabc, nabd, ap, ab, ap_ab,
        /*ap_ac, ap_ad,*/ bp_ab, /*, bp_ac, bp_ad*/
    );
    if res_ab.valid {
        return res_ab.proj;
    }

    let dabc = res_ab.du;
    let dabd = res_ab.dv;

    // Voronoï region of ac.
    // Substitutions (wrt. ab):
    //   b -> c
    //   c -> d
    //   d -> b
    //            let cp_ab = cp_ac - cp_bc;
    //            let cp_ad = cp_cd + cp_ac;
    let nacd = cross(ac, ad);
    let res_ac = check_edge(
        1, shape.a, shape.c, nacd, -nabc, ap, ac, ap_ac,
        /*ap_ad, ap_ab,*/ cp_ac, /*, cp_ad, cp_ab*/
    );
    if res_ac.valid {
        return res_ac.proj;
    }

    let dacd = res_ac.du;
    let dacb = res_ac.dv;

    // Voronoï region of ad.
    // Substitutions (wrt. ab):
    //   b -> d
    //   c -> b
    //   d -> c
    //            let dp_ac = dp_ad - dp_cd;
    //            let dp_ab = dp_ad - dp_bd;
    let res_ad = check_edge(
        2, shape.a, shape.d, -nabd, -nacd, ap, ad, ap_ad,
        /*ap_ab, ap_ac,*/ dp_ad, /*, dp_ab, dp_ac*/
    );
    if res_ad.valid {
        return res_ad.proj;
    }

    let dadb = res_ad.du;
    let dadc = res_ad.dv;

    // Voronoï region of bc.
    // Substitutions (wrt. ab):
    //   a -> b
    //   b -> c
    //   c -> a
    //            let cp_bd = cp_cd + cp_bc;
    let nbcd = cross(bc, bd);
    // NOTE: nabc = nbcd
    let res_bc = check_edge(
        3, shape.b, shape.c, nabc, nbcd, bp, bc, bp_bc,
        /*-bp_ab, bp_bd,*/ cp_bc, /*, -cp_ab, cp_bd*/
    );
    if res_bc.valid {
        return res_bc.proj;
    }

    let dbca = res_bc.du;
    let dbcd = res_bc.dv;

    // Voronoï region of bd.
    // Substitutions (wrt. ab):
    //   a -> b
    //   b -> d
    //   d -> a

    // let dp_bc = dp_bd - dp_cd;
    // NOTE: nbdc = -nbcd
    // NOTE: nbda = nabd
    let res_bd = check_edge(
        4, shape.b, shape.d, -nbcd, nabd, bp, bd, bp_bd,
        /*bp_bc, -bp_ab,*/ dp_bd, /*, dp_bc, -dp_ab*/
    );
    if res_bd.valid {
        return res_bd.proj;
    }

    let dbdc = res_bd.du;
    let dbda = res_bd.dv;

    // Voronoï region of cd.
    // Substitutions (wrt. ab):
    //   a -> c
    //   b -> d
    //   c -> a
    //   d -> b
    // NOTE: ncda = nacd
    // NOTE: ncdb = nbcd
    let res_cd = check_edge(
        5, shape.c, shape.d, nacd, nbcd, cp, cd, cp_cd,
        /*-cp_ac, -cp_bc,*/ dp_cd, /*, -dp_ac, -dp_bc*/
    );
    if res_cd.valid {
        return res_cd.proj;
    }

    let dcda = res_cd.du;
    let dcdb = res_cd.dv;

    // Face abc.
    let res_abc = check_face(
        0, shape.a, shape.b, shape.c, ap, bp, cp, ab, ac, ad, dabc, dbca,
        dacb,
        /*ap_ab, bp_ab, cp_ab,
        ap_ac, bp_ac, cp_ac*/
    );

    if res_abc.valid {
        return res_abc.proj;
    }

    // Face abd.
    let res_abd = check_face(
        1, shape.a, shape.b, shape.d, ap, bp, dp, ab, ad, ac, dadb, dabd,
        dbda,
        /*ap_ab, bp_ab, dp_ab,
        ap_ad, bp_ad, dp_ad*/
    );

    if res_abd.valid {
        return res_abd.proj;
    }

    // Face acd.
    let res_acd = check_face(
        2, shape.a, shape.c, shape.d, ap, cp, dp, ac, ad, ab, dacd, dcda,
        dadc,
        /*ap_ac, cp_ac, dp_ac,
        ap_ad, cp_ad, dp_ad*/
    );

    if res_acd.valid {
        return res_acd.proj;
    }

    // Face bcd.
    let res_bcd = check_face(
        3, shape.b, shape.c, shape.d, bp, cp, dp, bc, bd, -ab, dbcd, dcdb,
        dbdc,
        /*bp_bc, cp_bc, dp_bc,
        bp_bd, cp_bd, dp_bd*/
    );

    if res_bcd.valid {
        return res_bcd.proj;
    }

    return Proj::solid(pt);
}