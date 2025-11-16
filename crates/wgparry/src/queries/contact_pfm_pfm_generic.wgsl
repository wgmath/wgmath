#define_import_path wgparry::queries::contact_pfm_pfm_generic

#import wgparry::gjk::voronoi_simplex as VoronoiSimplex
#import wgparry::gjk::gjk as Gjk
#import wgparry::gjk::cso_point as CsoPoint
#import wgparry::epa as Epa
#import wgparry::cuboid as ShapeA
#import wgparry::cuboid as ShapeB
#import wgparry::polygonal_feature as PolygonalFeature
#import wgparry::contact_manifold as ContactManifold

#if DIM == 2
    #import wgebra::sim2 as Pose;
    const DIM: u32 = 2;
#else
    #import wgebra::sim3 as Pose;
    const DIM: u32 = 3;
#endif


fn contact_support_map_support_map(
    pose12: Transform,
    g1: ShapeA::Cuboid,
    g2: ShapeB::Cuboid,
    prediction: f32,
) -> Gjk::GjkResult {
    var dir = pose12.translation_scale.xyz;
    var dir_len_sq = dot(dir, dir);
    if dir_len_sq > CsoPoint::FLT_EPS * CsoPoint::FLT_EPS {
        dir /= sqrt(dir_len_sq);
    } else {
        dir = vec3(1.0, 0.0, 0.0);
    }

    let cso_point = Gjk::cso_point_from_shapes(pose12, g1, g2, dir);
    var simplex = VoronoiSimplex::init(cso_point);

    let cpts = Gjk::closest_points(pose12, g1, g2, prediction, true, &simplex);
    if cpts.status != Gjk::INTERSECTION {
        return cpts;
    }

    // The point is inside of the CSO: use the fallback algorithm
    let penetration = Epa::closest_points(pose12, g1, g2, simplex);
    if penetration.valid {
        return Gjk::gjk_result_closest_points(penetration.pt_a, penetration.pt_b, penetration.normal);
    }

    // Everything failed
    return Gjk::gjk_result_no_intersection(vec3(1.0, 0.0, 0.0));
}

fn contact_manifold_pfm_pfm(
    pose12: Transform,
    pfm1: ShapeA::Cuboid,
    border_radius1: f32,
    pfm2: ShapeB::Cuboid,
    border_radius2: f32,
    prediction: f32,
) -> ContactManifold::ContactManifold {
    let total_prediction = prediction + border_radius1 + border_radius2;
    let contact = contact_support_map_support_map(
        pose12,
        pfm1,
        pfm2,
        total_prediction,
    );

    switch contact.status {
        case Gjk::CLOSEST_POINTS: {
            let p1 = contact.a;
            let p2_1 = contact.b;
            var local_n1 = contact.dir;
            var local_n2 = Pose::invMulUnitVec(pose12, -local_n1);
            let dist = dot(p2_1 - p1, local_n1);

            let feature1 = ShapeA::support_face(pfm1, local_n1);
            let feature2 = ShapeA::support_face(pfm2, local_n2);
            var manifold = PolygonalFeature::contacts(
                pose12,
                Pose::inv(pose12),
                local_n1,
                local_n2,
                feature1,
                feature2,
                total_prediction,
                false
            );

            if manifold.len < ContactManifold::MAX_MANIFOLD_POINTS && (DIM == 3 || (DIM == 2 && manifold.len == 0u)) {
                let dist = dot(p2_1 - p1, local_n1);
                manifold.points_a[manifold.len] = ContactManifold::ContactPoint(p1, dist);
                manifold.len += 1;
            }

            // Adjust points to take the radius into account.
            if border_radius1 != 0.0 || border_radius2 != 0.0 {
                for (var i = 0u; i < manifold.len; i++) {
                    manifold.points_a[i].pt += local_n1 * border_radius1;
                    manifold.points_a[i].dist -= border_radius1 + border_radius2;
                }
            }

            manifold.normal_a = local_n1;
            return manifold;
        }
        default: { /* No collitions. */ }
    }

    return ContactManifold::ContactManifold();
}
