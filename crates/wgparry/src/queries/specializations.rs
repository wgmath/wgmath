use crate::substitute_aliases;

pub fn specialize_cylinder_cuboid(src: &str) -> String {
    substitute_aliases(src)
        .replace("ShapeA::Cuboid", "ShapeA::Cylinder")
        .replace("ShapeB::Cuboid", "ShapeB::Cuboid")
        .replace("cuboid as ShapeA", "cylinder as ShapeA")
        .replace("cuboid as ShapeB", "cuboid as ShapeB")
        .replace("contact_pfm_pfm_generic", "contact_cylinder_cuboid")
        .replace("wgparry::gjk::gjk", "wgparry::gjk::gjk_cylinder_cuboid")
        .replace("wgparry::epa", "wgparry::epa_cylinder_cuboid")
}

pub fn specialize_cylinder_cylinder(src: &str) -> String {
    substitute_aliases(src)
        .replace("ShapeA::Cuboid", "ShapeA::Cylinder")
        .replace("ShapeB::Cuboid", "ShapeB::Cylinder")
        .replace("cuboid as ShapeA", "cylinder as ShapeA")
        .replace("cuboid as ShapeB", "cylinder as ShapeB")
        .replace("contact_pfm_pfm_generic", "contact_cylinder_cylinder")
        .replace("wgparry::gjk::gjk", "wgparry::gjk::gjk_cylinder_cylinder")
        .replace("wgparry::epa", "wgparry::epa_cylinder_cylinder")
}

pub fn specialize_cylinder_cone(src: &str) -> String {
    substitute_aliases(src)
        .replace("ShapeA::Cuboid", "ShapeA::Cylinder")
        .replace("ShapeB::Cuboid", "ShapeB::Cone")
        .replace("cuboid as ShapeA", "cylinder as ShapeA")
        .replace("cuboid as ShapeB", "cone as ShapeB")
        .replace("contact_pfm_pfm_generic", "contact_cylinder_cone")
        .replace("wgparry::gjk::gjk", "wgparry::gjk::gjk_cylinder_cone")
        .replace("wgparry::epa", "wgparry::epa_cylinder_cone")
}

pub fn specialize_cuboid_cone(src: &str) -> String {
    substitute_aliases(src)
        .replace("ShapeA::Cuboid", "ShapeA::Cuboid")
        .replace("ShapeB::Cuboid", "ShapeB::Cone")
        .replace("cuboid as ShapeA", "cuboid as ShapeA")
        .replace("cuboid as ShapeB", "cone as ShapeB")
        .replace("contact_pfm_pfm_generic", "contact_cuboid_cone")
        .replace("wgparry::gjk::gjk", "wgparry::gjk::gjk_cuboid_cone")
        .replace("wgparry::epa", "wgparry::epa_cuboid_cone")
}

pub fn specialize_cone_cone(src: &str) -> String {
    substitute_aliases(src)
        .replace("ShapeA::Cuboid", "ShapeA::Cone")
        .replace("ShapeB::Cuboid", "ShapeB::Cone")
        .replace("cuboid as ShapeA", "cone as ShapeA")
        .replace("cuboid as ShapeB", "cone as ShapeB")
        .replace("contact_pfm_pfm_generic", "contact_cone_cone")
        .replace("wgparry::gjk::gjk", "wgparry::gjk::gjk_cone_cone")
        .replace("wgparry::epa", "wgparry::epa_cone_cone")
}
