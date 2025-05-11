#define_import_path wgparry::ray

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

/// The point on the ray at the given parameter `t`.
fn ptAt(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + ray.dir * t;
}