#![allow(dead_code)]

use std::cmp::Ordering;
use wgrapier_testbed2d::{SimulationState, Testbed};

mod balls2;
mod boxes2;
mod boxes_and_balls2;
mod joint_ball2;
mod joint_fixed2;
mod joint_prismatic2;
mod polyline2;
mod primitives2;
mod pyramid2;

fn demo_name_from_command_line() -> Option<String> {
    let mut args = std::env::args();

    while let Some(arg) = args.next() {
        if &arg[..] == "--example" {
            return args.next();
        }
    }

    None
}

#[cfg(target_arch = "wasm32")]
fn demo_name_from_url() -> Option<String> {
    None
    //    let window = stdweb::web::window();
    //    let hash = window.location()?.search().ok()?;
    //    Some(hash[1..].to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn demo_name_from_url() -> Option<String> {
    None
}

#[kiss3d::main]
pub async fn main() {
    let mut builders: Vec<(_, fn() -> SimulationState)> = vec![
        ("Balls", balls2::init_world),
        ("Boxes", boxes2::init_world),
        ("Boxes & balls", boxes_and_balls2::init_world),
        ("Pyramid", pyramid2::init_world),
        ("Primitives", primitives2::init_world),
        ("Polyline", polyline2::init_world),
        ("Joints (spherical)", joint_ball2::init_world),
        ("Joints (prismatic)", joint_prismatic2::init_world),
        ("Joints (fixed)", joint_fixed2::init_world),
    ];

    // Lexicographic sort, with stress tests moved at the end of the list.
    builders.sort_by(|a, b| match (a.0.starts_with('('), b.0.starts_with('(')) {
        (true, true) | (false, false) => a.0.cmp(b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
    });

    let testbed = Testbed::from_builders(builders);

    testbed.run().await
}
