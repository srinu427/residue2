pub mod output_info;
pub mod pipeline_info;
pub mod input_info;

pub struct PerFrameInfo {
    pub output_info: output_info::OutputInfo,
}

pub struct Renderer {
    pipeline_info: pipeline_info::PipelineInfo,
}