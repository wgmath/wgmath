#define_import_path wgcore::indirect

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}

struct AtomicDispatchIndirectArgs {
    x: atomic<u32>,
    y: atomic<u32>,
    z: atomic<u32>,
}

fn div_ceil(x: u32, y: u32) -> u32 {
    return (x + y - 1) / y;
}