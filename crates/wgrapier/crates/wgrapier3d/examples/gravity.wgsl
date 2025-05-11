#import wgrapier::body as Body;
#import wgebra::sim3 as Pose;

@group(0) @binding(0)
var<storage, read_write> mprops: array<Body::MassProperties>;
@group(0) @binding(1)
var<storage, read> local_mprops: array<Body::MassProperties>;
@group(0) @binding(2)
var<storage, read_write> poses: array<Pose::Sim3>;
@group(0) @binding(3)
var<storage, read_write> vels: array<Body::Velocity>;

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let gravity = Body::Force(vec3(0.0, -9.81, 0.0), vec3(0.0));
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < arrayLength(&poses); i += num_threads) {
        let new_vels = Body::integrateForces(mprops[i], vels[i], gravity, 0.016);
        let new_pose = Body::integrateVelocity(poses[i], new_vels, local_mprops[i].com, 0.016);
        let new_mprops = Body::updateMprops(new_pose, local_mprops[i]);

        mprops[i] = new_mprops;
        vels[i] = new_vels;
        poses[i] = new_pose;
    }
}
