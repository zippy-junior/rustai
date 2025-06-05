use std::ops;

#[derive(Debug)]
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

impl<T, const LH_ROWS: usize, const LH_COLS: usize, const RH_ROWS: usize, const RH_COLS: usize> ops::Mul<Tensor<T, RH_ROWS, RH_COLS>> for Tensor<T, LH_ROWS, LH_COLS>
where
    T: ops::Add<Output = T> + Default + ops::Mul<Output = T> + Copy + std::fmt::Debug,  // Element type must support addition and be copyable
{
    type Output = Tensor<T, LH_ROWS, RH_COLS>;  // Result of addition is another Tensor with same dimensions

    fn mul(self, rhs: Tensor<T, RH_ROWS, RH_COLS>) -> Tensor<T, LH_ROWS, RH_COLS> {
        assert!(LH_COLS == RH_ROWS, 
            "Size of matrices are not compatible
            for multiplication (left hand matrix columns
            are not equal to right hand matrix rows)");

        let mut result = Tensor::<T, LH_ROWS, RH_COLS>::new();
        
        for i in 0..LH_ROWS {
            for j in 0..RH_COLS {
                let mut sum = T::default();
                for k in 0..RH_ROWS {
                    sum = sum + (self.data[i][k] * rhs.data[k][j]);
                    dbg!(sum);
                }
                result.data[i][j] = sum;
            }
        }
        
        result
    }
}