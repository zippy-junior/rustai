pub mod layer;
pub mod tensor;
mod test_data;
use crate::layer::{DenseLayer, Layer};
use test_data::test_data;


// #[path = "./tensor/tensor.rs"]


fn main() {
    // let inputs: tensor::Tensor<f32, 3, 4> = Tensor::from_data([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]);
    // let weights: tensor::Tensor<f32, 3, 4> = Tensor::from_data([[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]);
    // let biases: tensor::Tensor<f32, 1, 3> = Tensor::from_data([[2.0, 3.0, 0.5]]);
    // let res = inputs * weights.transpose() + biases.broadcast::<3, 3>();
    // println!("{:?}", res);
    type MyDenseLayer = DenseLayer<2, 3, 300>;

    let layer= MyDenseLayer::new();
    let res = layer.forward(test_data());
    println!("{:?}", res);
}