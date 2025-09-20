#import wgparry::shape as Shape;
#import wgparry::contact as Contact;
#import wgebra::sim3 as Pose;
#import wgcore::indirect as Indirect;




@group(0) @binding(0)
var<storage, read_write> collision_pairs: array<vec2<u32>>;
@group(0) @binding(1)
var<storage, read_write> collision_pairs_len: u32;
@group(0) @binding(2)
var<storage, read> poses: array<Pose::Sim3>;
@group(0) @binding(3)
var<storage, read> shapes: array<Shape::Shape>;
@group(0) @binding(4)
var<storage, read_write> contacts: array<Contact::IndexedContact>;
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
        let shape1 = Shape::to_ball(shapes[pair.x]);
        let shape2 = Shape::to_ball(shapes[pair.y]);
        let pose12 = Pose::invMul(pose1, pose2);
        let contact = Contact::ball_ball(pose12, shape1, shape2);
        let prediction = 2.0e-3; // TODO: make the prediciton configurable.

        if contact.dist < prediction {
            let target_contact_index = atomicAdd(&contacts_len, 1u);
            contacts[target_contact_index] = Contact::IndexedContact(contact, pair);
        }
    }
}
