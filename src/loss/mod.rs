use rustai::tensor::Tensor;

pub struct CrossEntropyLoss<const N_INPUTS: usize, const BATCH_SIZE: usize> {}

pub enum Targets<const N_INPUTS: usize, const BATCH_SIZE: usize>{
    categorical(Tensor<f32, N_INPUTS, BATCH_SIZE>),
    onehot(Tensor<f32, 1, BATCH_SIZE>)
}

pub trait Loss<const N_INPUTS: usize, const BATCH_SIZE: usize>{
    fn forward(inputs: Tensor<f32, N_INPUTS, BATCH_SIZE>, targets: Targets<N_INPUTS, BATCH_SIZE>) {}
}

impl <const N_INPUTS: usize, const BATCH_SIZE: usize> Loss<N_INPUTS, BATCH_SIZE> for CrossEntropyLoss<N_INPUTS, BATCH_SIZE> {
    fn forward(inputs: Tensor<f32, N_INPUTS, BATCH_SIZE>, targets: Targets<N_INPUTS, BATCH_SIZE>) {
        match targets {
            Targets::categorical(t) => {
                
                // let correct_confidences = inputs[ ​range​(​len​(softmax_outputs)), class_targets]
            },
            Targets::onehot(t) => {

            }
        }
    }
}