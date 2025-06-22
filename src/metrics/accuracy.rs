use crate::{metrics::Targets, tensor::{Tensor, TensorConvert}};


pub struct Accuracy<const BATCH_SIZE: usize, const N_INPUTS: usize> {}

impl<const BATCH_SIZE: usize, const N_INPUTS: usize> Accuracy<BATCH_SIZE, N_INPUTS> {
    pub fn calculate(&self, inputs: Tensor<f32, BATCH_SIZE, N_INPUTS>, targets: super::Targets<BATCH_SIZE, N_INPUTS>) -> Option<f32> 
    {
        let predictions = inputs.argmax(crate::tensor::Axis::Row)?.unwrap_row();
        let class_targets = match targets {
            Targets::onehot(t) => {
                t
            },
        }.argmax(crate::tensor::Axis::Row)?.unwrap_row();
        let converted: Tensor<f32, BATCH_SIZE, 1> = predictions.eq(class_targets).convert();
        Some(converted.mean())
    }
}