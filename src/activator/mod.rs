pub mod relu;
pub mod softmax;

use crate::tensor::Tensor;

pub trait Activator<const BATCH_SIZE: usize, const N_INPUTS: usize> {
    fn forward(self, inputs: &Tensor<f32, BATCH_SIZE, N_INPUTS>) -> Tensor<f32, BATCH_SIZE, N_INPUTS>;
}
