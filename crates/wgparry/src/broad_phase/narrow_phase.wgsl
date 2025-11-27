//! Narrow Phase Contact Generation
//!
//! Computes contact manifolds from the collision pairs detected by the the broad phase.
//! It performs shape-specific narrow-phase collision detection for each potentially colliding pair.
//!
//! Pipeline:
//! 1. reset: Clears contact counter.
//! 2. main: Processes each collision pair, computes contacts, filters by distance.
//! 3. init_indirect_args: Prepares indirect dispatch for constraint solver.
//!
//! Buffers:
//! - Input: collision_pairs, collision_pairs_len, poses, shapes
//! - Output: contacts (IndexedManifold array), contacts_len (atomic counter)
//!
//! Supported shape pairs:
//! - Ball-Ball: Analytical distance computation
//! - Ball-Cuboid / Cuboid-Ball: Point projection
//! - Cuboid-Cuboid: SAT + feature clipping

#import wgparry::shape as Shape;
#import wgparry::contact as Contact;
#import wgparry::contact_manifold as Manifold;
#import wgcore::indirect as Indirect;

#if DIM == 2
    #import wgebra::sim2 as Pose;
#else
    #import wgebra::sim3 as Pose;
#endif

@group(0) @binding(0)
var<storage, read_write> collision_pairs: array<vec2<u32>>;
@group(0) @binding(1)
var<storage, read_write> collision_pairs_len: u32;
@group(0) @binding(2)
var<storage, read> poses: array<Transform>;
@group(0) @binding(3)
var<storage, read> shapes: array<Shape::Shape>;
@group(0) @binding(4)
var<storage, read_write> contacts: array<Contact::IndexedManifold>;
@group(0) @binding(5)
var<storage, read_write> contacts_len: atomic<u32>;
@group(0) @binding(6)
var<storage, read_write> contacts_len_indirect_args: Indirect::DispatchIndirectArgs;

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(1, 1, 1)
fn reset() {
    contacts_len = 0u;
}

@compute @workgroup_size(1, 1, 1)
fn init_indirect_args() {
    contacts_len_indirect_args = Indirect::DispatchIndirectArgs(Indirect::div_ceil(contacts_len, WORKGROUP_SIZE), 1, 1);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < collision_pairs_len; i += num_threads) {
        let pair = collision_pairs[i];
        let pose1 = poses[pair.x];
        let pose2 = poses[pair.y];
        let shape_ty1 = Shape::shape_type(shapes[pair.x]);
        let shape_ty2 = Shape::shape_type(shapes[pair.y]);
        var manifold = Manifold::ContactManifold();
        let pose12 = Pose::invMul(pose1, pose2);
        let prediction = 2.0e-3; // TODO: make the prediciton configurable.
        var checked = false;

        // Ball - Convex
        if shape_ty1 == Shape::SHAPE_TYPE_BALL {
            switch shape_ty2 {
                case Shape::SHAPE_TYPE_BALL: {
                    let shape1 = Shape::to_ball(shapes[pair.x]);
                    let shape2 = Shape::to_ball(shapes[pair.y]);
                    manifold = Contact::ball_ball(pose12, shape1, shape2);
                    checked = true;
                }
                case Shape::SHAPE_TYPE_CUBOID, Shape::SHAPE_TYPE_CAPSULE,
                    Shape::SHAPE_TYPE_CONE, Shape::SHAPE_TYPE_CYLINDER: {
                    let shape1 = Shape::to_ball(shapes[pair.x]);
                    manifold = Contact::ball_convex(pose12, shape1, shapes[pair.y]);
                    checked = true;
                }
                default: {
                }
            }
        }

        // Convex - Ball
        if !checked && shape_ty2 == Shape::SHAPE_TYPE_BALL {
            switch shape_ty1 {
                case Shape::SHAPE_TYPE_CUBOID, Shape::SHAPE_TYPE_CAPSULE,
                    Shape::SHAPE_TYPE_CONE, Shape::SHAPE_TYPE_CYLINDER: {
                    let shape2 = Shape::to_ball(shapes[pair.y]);
                    manifold = Contact::convex_ball(pose12, shapes[pair.x], shape2);
                    checked = true;
                }
                default: {
                }
            }
        }

        // Cuboid - cuboid
        if !checked && shape_ty1 == Shape::SHAPE_TYPE_CUBOID && shape_ty2 == Shape::SHAPE_TYPE_CUBOID {
            let shape1 = Shape::to_cuboid(shapes[pair.x]);
            let shape2 = Shape::to_cuboid(shapes[pair.y]);
            manifold = Contact::cuboid_cuboid(pose12, shape1, shape2, prediction);
            checked = true;
        }

        // Pfm - Pfm
        if !checked {
            let sub1 = Shape::pfm_subshape(shapes[pair.x]);
            let sub2 = Shape::pfm_subshape(shapes[pair.y]);

            if sub1.valid && sub2.valid {
                manifold = Contact::pfm_pfm(pose12, sub1.shape, sub1.thickness, sub2.shape, sub2.thickness, prediction);
            }
        }

        if manifold.len > 0 && manifold.points_a[0].dist < prediction {
            let target_contact_index = atomicAdd(&contacts_len, 1u);
            contacts[target_contact_index] = Contact::IndexedManifold(manifold, pair);
        }
    }
}
