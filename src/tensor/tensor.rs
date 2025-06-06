use std::ops;

#[derive(Debug)]
#[derive(Clone, Copy)]
pub struct Tensor<T, const ROWS: usize, const COLS: usize> {
    data: [[T; COLS]; ROWS]
}

impl<T, const ROWS: usize, const COLS: usize> Tensor<T, ROWS, COLS> {
    pub fn new() -> Self
    where
        T: Default + Copy,
    {
        Tensor {
            data: [[T::default(); COLS]; ROWS]
        }
    }

    pub fn fill(val: T) -> Self
    where
        T: Copy,
    {
        Tensor {
            data: [[val; COLS]; ROWS]
        }
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

impl<T, const ROWS: usize, const COLS: usize> ops::Add<T> for Tensor<T, ROWS, COLS>
where
    T: ops::Add<Output = T> + Copy,  // Element type must support addition and be copyable
{
    type Output = Self;  // Result of addition is another Tensor with same dimensions

    fn add(self, scalar: T) -> Tensor<T, ROWS, COLS> {
        let mut result = self;  // Copy self (requires T: Copy)
        
        // Element-wise addition
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = result.data[i][j] + scalar;
            }
        }
        
        result
    }
}

impl<T, const LH_ROWS: usize, const LH_COLS: usize, const RH_COLS: usize> ops::Mul<Tensor<T, LH_COLS, RH_COLS>> for Tensor<T, LH_ROWS, LH_COLS>
where
    T: ops::Add<Output = T> + Default + ops::Mul<Output = T> + Copy + std::fmt::Debug,  // Element type must support addition and be copyable
{
    type Output = Tensor<T, LH_ROWS, RH_COLS>;  // Result of addition is another Tensor with same dimensions

    fn mul(self, rhs: Tensor<T, LH_COLS, RH_COLS>) -> Tensor<T, LH_ROWS, RH_COLS> {

        let mut result = Tensor::<T, LH_ROWS, RH_COLS>::new();
        
        for i in 0..LH_ROWS {
            for j in 0..RH_COLS {
                let mut sum = T::default();
                for k in 0..LH_COLS {
                    sum = sum + (self.data[i][k] * rhs.data[k][j]);
                }
                result.data[i][j] = sum;
            }
        }
        
        result
    }
}

impl<T, const ROWS: usize, const COLS: usize> ops::Mul<T> for Tensor<T, ROWS, COLS>
where
    T: ops::Add<Output = T> + Default + ops::Mul<Output = T> + Copy,  // Element type must support addition and be copyable
{
    type Output = Tensor<T, ROWS, COLS>;  // Result of addition is another Tensor with same dimensions

    fn mul(self, scalar: T) -> Tensor<T, ROWS, COLS> {
        let mut result = self;
        
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = result.data[i][j] * scalar;
            }
        }
                
        result
    }
}