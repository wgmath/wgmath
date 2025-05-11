#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::ray as Ray
#import wgparry::projection as Proj

#define_import_path wgparry::segment

struct Segment {
    a: Vector,
    b: Vector,
}

// TODO: implement the other projection functions
fn projectLocalPoint(seg: Segment, pt: Vector) -> Vector {
    let ab = seg.b - seg.a;
    let ap = pt - seg.a;
    let ab_ap = dot(ab, ap);
    let sqnab = dot(ab, ab);

    // PERF: would it be faster to do a bunch of `select` instead of `if`?
    if ab_ap <= 0.0 {
        // Voronoï region of vertex 'a'.
        return seg.a;
    } else if ab_ap >= sqnab {
        // Voronoï region of vertex 'b'.
        return seg.b;
    } else {
        // Voronoï region of the segment interior.
        let u = ab_ap / sqnab;
        return seg.a + ab * u;
    }
}
