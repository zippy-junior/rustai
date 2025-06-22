use crate::tensor::{Tensor, TensorIndex};


pub struct CrossEntropyLoss<const BATCH_SIZE: usize, const N_INPUTS: usize> {}

pub trait Loss<const BATCH_SIZE: usize, const N_INPUTS: usize>{
    fn forward(&self, inputs: Tensor<f32, BATCH_SIZE, N_INPUTS>, targets: super::Targets<BATCH_SIZE, N_INPUTS>) -> f32;
}

impl <const BATCH_SIZE: usize, const N_INPUTS: usize> Loss<BATCH_SIZE, N_INPUTS> for CrossEntropyLoss<BATCH_SIZE, N_INPUTS> {
    fn forward(&self, inputs: Tensor<f32, BATCH_SIZE, N_INPUTS>, targets: super::Targets<BATCH_SIZE, N_INPUTS>) -> f32 {
        // TODO not clone the inputs for efficiency
        let mut clipped_inputs = inputs.clone();
        clipped_inputs.apply(|el| {
            if *el == 1 as f32 {
                1 as f32 - f32::MIN
            } else if *el == 0 as f32 {
                0 as f32 + f32::MIN
            } else {
                *el
            }
        });
        match targets {
            // Targets::categorical(t) => {
                
            // },
            super::Targets::onehot(t) => {
                // TODO Process result
                let mut masked_result = clipped_inputs.index_cols(TensorIndex::Mask(t)).expect("Error while indexing with onehot targets");
                masked_result.apply(|el| -el.ln());
                masked_result.mean()
            }
        }
    }
}