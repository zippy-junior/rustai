pub mod layer;
pub mod tensor;
pub mod activator;
pub mod loss;
mod test_data;

use crate::activator::relu::ReLU;
use crate::activator::softmax::Softmax;
use crate::layer::{DenseLayer, Layer};
// use crate::tensor::Tensor;
use crate::activator::{Activator};
use crate::loss::{CrossEntropyLoss, Loss, Targets};
// use crate::tensor::AxisRes;
use test_data::{test_data, test_targets};

fn main() {
    // let weights: tensor::Tensor<f32, 3, 4> = Tensor::from_data([[0.0, 0.0, 0.0, 0.0],[0.0, -0.91, 0.26, -0.5],[0.0, -0.27, 0.17, 0.87]]);
    // let biases: tensor::Tensor<f32, 1, 3> = Tensor::from_data([[2.0, 3.0, 0.5]]);
    // let res = inputs * weights.transpose() + biases.broadcast::<3, 3>();
    // println!("{:?}", weights.all(tensor::Axis::Col, |&x| {x == 0.0}));
    // println!("{:?}", weights.transpose());


    let layer_1 = DenseLayer::<2, 3, 300>::new();
    let res = layer_1.forward(test_data());
    let activator: ReLU<300, 3> = ReLU{};
    let activator_res = activator.forward(&res);
    let layer_2 = DenseLayer::<3, 3, 300>::new();
    let res_2 = layer_2.forward(activator_res);
    let activator_2: Softmax<300, 3> = Softmax{};
    let activator_res_2 = activator_2.forward(&res_2);
    let loss: CrossEntropyLoss<300, 3> = CrossEntropyLoss {};
    let loss_res = loss.forward(activator_res_2, Targets::onehot(test_targets()));
    println!("{:?}", loss_res);
}