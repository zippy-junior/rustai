use crate::tensor::Tensor;

pub mod loss;
pub mod accuracy;

pub enum Targets<const BATCH_SIZE: usize, const N_INPUTS: usize>{
    onehot(Tensor<usize, BATCH_SIZE, N_INPUTS>)
    // categorical(Tensor<usize, 1, BATCH_SIZE>)
}