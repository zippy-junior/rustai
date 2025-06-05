use std::ops;

#[derive(Debug)]
pub struct Tensor<T, const ROWS: usize, const COLUMNS: usize> {
    data: [[T; ROWS]; COLUMNS]
}

impl<T, const ROWS: usize, const COLUMNS: usize> Tensor<T, ROWS, COLUMNS> {
    pub fn new() -> Self
    where
        T: Default + Copy,
    {
        Tensor {
            data: [[T::default(); ROWS]; COLUMNS]
        }
    }

    pub fn fill(val: T) -> Self
    where
        T: Default + Copy,
    {
        Tensor {
            data: [[val; ROWS]; COLUMNS]
        }
    }

    pub fn add(self, _rhs: Self) -> Tensor<T, ROWS, COLUMNS>
    where
        T: ops::Add<Output = T> + Copy,
    {
        let mut result = self;  // Copy the first tensor
        
        // Perform element-wise addition
        for i in 0..ROWS {
            for j in 0..COLUMNS {
                result.data[i][j] = result.data[i][j] + _rhs.data[i][j];
            }
        }
        
        result
    }
}

impl<T, const ROWS: usize, const COLS: usize> ops::Add for Tensor<T, ROWS, COLS>
where
    T: ops::Add<Output = T> + Copy,  // Element type must support addition and be copyable
{
    type Output = Self;  // Result of addition is another Tensor with same dimensions

    fn add(self, rhs: Self) -> Tensor<T, ROWS, COLS> {
        let mut result = self;  // Copy self (requires T: Copy)
        
        // Element-wise addition
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = result.data[i][j] + rhs.data[i][j];
            }
        }
        
        result
    }
}