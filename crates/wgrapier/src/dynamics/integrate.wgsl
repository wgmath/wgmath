#define_import_path wgrapier::integrate

#import wgrapier::body as Body;

#if DIM == 2
    #import wgebra::sim2 as Pose;
#else
    #import wgebra::sim3 as Pose;
#endif

@group(0) @binding(0)
var<storage, read_write> mprops: array<Body::WorldMassProperties>;
@group(0) @binding(1)
var<storage, read> local_mprops: array<Body::LocalMassProperties>;
#if DIM == 2
@group(0) @binding(2)
var<storage, read_write> poses: array<Pose::Sim2>;
#else
@group(0) @binding(2)
var<storage, read_write> poses: array<Pose::Sim3>;
#endif
@group(0) @binding(3)
var<storage, read_write> vels: array<Body::Velocity>;

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn integrate(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    for (var i = invocation_id.x; i < arrayLength(&poses); i += num_threads) {
        // TODO: get dt from somewhere
        let new_pose = Body::integrateVelocity(poses[i], vels[i], local_mprops[i].com, 0.0016);
        let new_mprops = Body::updateMprops(new_pose, local_mprops[i]);

        mprops[i] = new_mprops;
        poses[i] = new_pose;
    }
}
