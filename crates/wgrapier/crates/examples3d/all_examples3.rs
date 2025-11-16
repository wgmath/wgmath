#![allow(dead_code)]

use inflector::Inflector;

use std::cmp::Ordering;
use wgrapier_testbed3d::{SimulationState, Testbed};

mod balls3;
mod boxes3;
mod boxes_and_balls3;
mod joint_ball3;
mod joint_fixed3;
mod joint_prismatic3;
mod joint_revolute3;
mod keva3;
mod many_pyramids3;
mod pyramid3;

enum Command {
    Run(String),
    List,
    RunAll,
}

fn parse_command_line() -> Command {
    let mut args = std::env::args();

    while let Some(arg) = args.next() {
        if &arg[..] == "--example" {
            return Command::Run(args.next().unwrap_or_default());
        } else if &arg[..] == "--list" {
            return Command::List;
        }
    }

    Command::RunAll
}

#[allow(clippy::type_complexity)]
pub fn demo_builders() -> Vec<(&'static str, fn() -> SimulationState)> {
    let mut builders: Vec<(_, fn() -> SimulationState)> = vec![
        ("Balls", balls3::init_world),
        ("Boxes", boxes3::init_world),
        ("Boxes & balls", boxes_and_balls3::init_world),
        ("Pyramid", pyramid3::init_world),
        ("Many pyramids", many_pyramids3::init_world),
        ("Keva tower", keva3::init_world),
        ("Joints (Spherical)", joint_ball3::init_world),
        ("Joints (Fixed)", joint_fixed3::init_world),
        ("Joints (Prismatic)", joint_prismatic3::init_world),
        ("Joints (Revolute)", joint_revolute3::init_world),
    ];

    // Lexicographic sort, with stress tests moved at the end of the list.
    builders.sort_by(|a, b| match (a.0.starts_with('('), b.0.starts_with('(')) {
        (true, true) | (false, false) => a.0.cmp(b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
    });
    builders
}

#[kiss3d::main]
pub async fn main() {
    let command = parse_command_line();
    let builders = demo_builders();

    match command {
        Command::Run(demo) => {
            if let Some(i) = builders
                .iter()
                .position(|builder| builder.0.to_camel_case().as_str() == demo.as_str())
            {
                Testbed::from_builders(vec![builders[i]]).run().await
            } else {
                eprintln!("Invalid example to run provided: '{demo}'");
            }
        }
        Command::RunAll => Testbed::from_builders(builders).run().await,
        Command::List => {
            for builder in &builders {
                println!("{}", builder.0.to_camel_case())
            }
        }
    }
}
