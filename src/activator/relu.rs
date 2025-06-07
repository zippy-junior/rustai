use crate::tensor::Tensor;

pub struct ReLU<const BATCH_SIZE: usize, const N_INPUTS: usize>;

impl<const N_INPUTS: usize, const BATCH_SIZE: usize> super::Activator<BATCH_SIZE, N_INPUTS> for ReLU<BATCH_SIZE, N_INPUTS> {
    fn forward(self, inputs: &Tensor<f32, BATCH_SIZE, N_INPUTS>) -> Tensor<f32, BATCH_SIZE, N_INPUTS> {
        let mut res = inputs.clone();
        res.apply(|el| {
            match el < &0.0 {
                true => 0.0,
                false => *el
            }
        });
        res
    }
}