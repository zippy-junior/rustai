use crate::tensor::Tensor;

#[path = "./tensor/tensor.rs"]
mod tensor;

fn main() {
    let mut tensor: tensor::Tensor<f32, 10, 10> = Tensor::fill(0.2);
    tensor = Tensor::fill(0.5);
    let tensor_2: tensor::Tensor<f32, 10, 10> = Tensor::fill(0.5);
    let res = tensor + tensor_2;
    println!("{:?}", res);
}