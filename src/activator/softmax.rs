use crate::{activator::Activator, tensor::Tensor};
use crate::tensor::Axis;

const E: f32 = 2.71828182846;

pub struct Softmax<const BATCH_SIZE: usize, const N_INPUTS: usize>;

impl<const N_INPUTS: usize, const BATCH_SIZE: usize> super::Activator<BATCH_SIZE, N_INPUTS> for Softmax<BATCH_SIZE, N_INPUTS>  {
    fn forward(self, inputs: &Tensor<f32, BATCH_SIZE, N_INPUTS>) -> Tensor<f32, BATCH_SIZE, N_INPUTS> {
        let mut res = inputs.clone();
        let max_col = res.max_axis(Axis::Row).unwrap_row();
        res = res - max_col.broadcast::<BATCH_SIZE, N_INPUTS>();
        res.apply(|el| { E ** el });
        let exp_sum = res.sum_axis(Axis::Row).unwrap_row();
        res / exp_sum.broadcast::<BATCH_SIZE, N_INPUTS>()
    }
}