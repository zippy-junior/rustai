use crate::tensor::Tensor;


pub trait Layer<const N_INPUTS: usize, const N_NEURONS: usize, const BATCH_SIZE: usize> {
    fn forward(self, inputs: Tensor<f32, BATCH_SIZE, N_INPUTS>) -> Tensor<f32, BATCH_SIZE, N_NEURONS>;
    fn new() -> DenseLayer<N_INPUTS, N_NEURONS, BATCH_SIZE>;
}

pub struct DenseLayer<const N_INPUTS: usize, const N_NEURONS: usize, const BATCH_SIZE: usize> {
    weights: Tensor<f32, N_INPUTS, N_NEURONS>,
    biases: Tensor<f32, 1, N_NEURONS>
}

// For now we use concrete f32 for dense layer. No need for generic
impl <const N_INPUTS: usize, const N_NEURONS: usize, const BATCH_SIZE: usize> Layer<N_INPUTS, N_NEURONS, BATCH_SIZE> for DenseLayer<N_INPUTS, N_NEURONS, BATCH_SIZE> {
    fn new() -> DenseLayer<N_INPUTS, N_NEURONS, BATCH_SIZE> {
        let weights: Tensor<f32, N_INPUTS, N_NEURONS> = Tensor::rand_fill() * 0.01;
        let biases: Tensor<f32, 1, N_NEURONS> = Tensor::new();
        DenseLayer { weights: weights, biases: biases }
    }

    fn forward(self, inputs: Tensor<f32, BATCH_SIZE, N_INPUTS>) -> Tensor<f32, BATCH_SIZE, N_NEURONS> {
        inputs * self.weights + self.biases.broadcast::<BATCH_SIZE, N_NEURONS>()
    }
}