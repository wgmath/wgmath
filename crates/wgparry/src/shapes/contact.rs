use crate::ball::WgBall;
use wgcore::Shader;
use wgebra::WgSim3;

#[derive(Shader)]
#[shader(derive(WgSim3, WgBall, WgProjection), src = "contact.wgsl")]
pub struct WgContact;

wgcore::test_shader_compilation!(WgContact);
