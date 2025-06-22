use num_traits::{cast::FromPrimitive, Num};
use std::{fmt::Display, iter::{zip, Sum}, ops};
use rand;


#[derive(Debug)]
#[derive(Clone, Copy)]
pub struct Tensor<T, const ROWS: usize, const COLS: usize> {
    data: [[T; COLS]; ROWS]
}

pub enum Axis {
    Row,
    Col
}

#[derive(Debug)]
pub enum AxisRes<T, const ROWS: usize, const COLS: usize> {
    Row(Tensor<T, ROWS, 1>),
    Col(Tensor<T, 1, COLS>)
}

pub enum TensorIndex<const ROWS: usize, const COLS: usize> {
    Scalar(usize),
    Mask(Tensor<usize, ROWS, COLS>)
}

impl<T, const ROWS: usize, const COLS: usize> AxisRes<T, ROWS, COLS> {
    pub fn unwrap_row(self) -> Tensor<T, ROWS, 1> {
        match self {
            AxisRes::Row(tensor) => tensor,
            _ => panic!("Called unwrap_row on AxisRes::Col"),
        }
    }
    
    pub fn unwrap_col(self) -> Tensor<T, 1, COLS> {
        match self {
            AxisRes::Col(tensor) => tensor,
            _ => panic!("Called unwrap_col on AxisRes::Row"),
        }
    }
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

    pub fn from_data(data: [[T; COLS]; ROWS]) -> Self
    where
        T: Default + Copy,
    {
        Tensor {
            data: data
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

    pub fn rand_fill() -> Self 
    where
        T: Copy,
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        Tensor {
            data: [[rand::random::<T>(); COLS]; ROWS]
        }
    }

    pub fn dot(self, rhs: Tensor<T, COLS, 1>) -> T
    where 
        T: Default + ops::Mul<Output = T> + ops::Add<Output = T> + Copy
    {
        let mut res: T = T::default();
        for i in 0..COLS {
            res = res + (self.data[0][i] * rhs.data[i][0])
        }
        res
    }

    pub fn transpose(self) -> Tensor<T, COLS, ROWS>
    where 
        T: Default + Copy
    {
        let mut res: Tensor<T, COLS, ROWS> = Tensor::new();

        for i in 0..COLS {
            for j in 0..ROWS {
            res.data[i][j] = self.data[j][i];
            }
        }

        res
    }

    pub fn broadcast<const T_ROWS: usize, const T_COLS: usize>(&self) -> Tensor<T, T_ROWS, T_COLS>
    where
        T: Copy,
    {
        // Check broadcasting rules
        assert!(
            ROWS == T_ROWS || ROWS == 1,
            "Row dimension must match or be 1 for broadcasting"
        );
        assert!(
            COLS == T_COLS || COLS == 1,
            "Column dimension must match or be 1 for broadcasting"
        );

        let mut result = Tensor {
            data: [[self.data[0][0]; T_COLS]; T_ROWS],
        };

        for i in 0..T_ROWS {
            for j in 0..T_COLS {
                // Determine source indices with broadcasting rules
                let src_i = if ROWS == 1 { 0 } else { i };
                let src_j = if COLS == 1 { 0 } else { j };
                
                result.data[i][j] = self.data[src_i][src_j].clone();
            }
        }

        result
    }

    pub fn apply<F>(&mut self, func: F)
    where
        F: Fn(&T) -> T
    {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i][j] = func(&self.data[i][j])
            }
        }
    }

    pub fn sum(&self) -> T
    where
        T: Sum + Copy,
    {
        self.data.iter().flatten().copied().sum()
    }
    
    pub fn sum_axis(&self, axis: Axis) -> AxisRes<T, ROWS, COLS> 
    where 
        T: Default + Copy + std::ops::Add<Output = T>
    {
        match axis {
            Axis::Row => {
                // Sum each row (result will be column vector)
                let mut result = Tensor::<T, ROWS, 1>::new();
                for i in 0..ROWS {
                    let mut sum = T::default();
                    for j in 0..COLS {
                        sum = sum + self.data[i][j];
                    }
                    result.data[i][0] = sum;
                }
                AxisRes::Row(result)
            }
            Axis::Col => {
                // Sum each column (result will be row vector)
                let mut result = Tensor::<T, 1, COLS>::new();
                for j in 0..COLS {
                    let mut sum = T::default();
                    for i in 0..ROWS {
                        sum = sum + self.data[i][j];
                    }
                    result.data[0][j] = sum;
                }
                AxisRes::Col(result)
            }
        }
    }

    pub fn max(&self) -> T 
    where 
        T: Default + std::cmp::PartialOrd + Copy
    {
        let mut max_res: T = self.data[0][0];
        for i in 0..ROWS {
            for j in 0..COLS {
                if self.data[i][j] > max_res {
                    max_res = self.data[i][j];
                } else { 
                    continue;
                }
            }
        }
        max_res
    }

    pub fn max_axis(&self, axis: Axis) -> AxisRes<T, ROWS, COLS> 
    where 
        T: Default + Copy + std::ops::Add<Output = T>
    {
        match axis {
            Axis::Row => {
                // Sum each row (result will be column vector)
                let mut result = Tensor::<T, ROWS, 1>::new();
                for i in 0..ROWS {
                    let mut sum = T::default();
                    for j in 0..COLS {
                        sum = sum + self.data[i][j];
                    }
                    result.data[i][0] = sum;
                }
                AxisRes::Row(result)
            }
            Axis::Col => {
                // Sum each column (result will be row vector)
                let mut result = Tensor::<T, 1, COLS>::new();
                for j in 0..COLS {
                    let mut sum = T::default();
                    for i in 0..ROWS {
                        sum = sum + self.data[i][j];
                    }
                    result.data[0][j] = sum;
                }
                AxisRes::Col(result)
            }
        }
    }
    pub fn any<F>(&self, axis: Axis, mut f: F) -> AxisRes<bool, ROWS, COLS> 
    where
        F: FnMut(&T) -> bool,
        T: Default + Copy
    {
            match axis {
            Axis::Row => {
                let mut any_row = Tensor::<bool, ROWS, 1>::new();
                for i in 0..ROWS {
                    let mut row: bool = false;
                    for j in 0..COLS {
                        if f(&self.data[i][j]) {
                            row = true;
                        }
                    }
                    any_row.data[i] = [row];    
                }
                AxisRes::Row(any_row)
            }
            Axis::Col => {
                let transposed = self.transpose();
                let mut any_col = Tensor::<bool, 1, COLS>::new();
                for j in 0..COLS {
                    let mut col: bool = false;
                    for i in 0..ROWS {
                        if f(&transposed.data[j][i]) {
                            col = true;
                        }
                    }
                    any_col.data[0][j] = col;    
                }
                AxisRes::Col(any_col)
            }
        }
    }
    pub fn all<F>(&self, axis: Axis, mut f: F) -> AxisRes<bool, ROWS, COLS> 
    where
        F: FnMut(&T) -> bool,
        T: Default + Copy
    {
            match axis {
            Axis::Row => {
                let mut any_row = Tensor::<bool, ROWS, 1>::new();
                for i in 0..ROWS {
                    let mut row: bool = true;
                    for j in 0..COLS {
                        if !f(&self.data[i][j]) {
                            row = false;
                        }
                    }
                    any_row.data[i] = [row];    
                }
                AxisRes::Row(any_row)
            }
            Axis::Col => {
                let transposed = self.transpose();
                let mut any_col = Tensor::<bool, 1, COLS>::new();
                for j in 0..COLS {
                    let mut col: bool = true;
                    for i in 0..ROWS {
                        if !f(&transposed.data[j][i]) {
                            col = false;
                        }
                    }
                    any_col.data[0][j] = col;    
                }
                AxisRes::Col(any_col)
            }
        }
    }
    pub fn index_cols(&self, idx: TensorIndex<ROWS, COLS>) -> Result<Tensor<T, ROWS, 1>, String>
    where
        T: Default + Copy + Display
    {
        let mut res: Tensor<T, ROWS, 1> = Tensor::new();
        match idx {
            TensorIndex::Scalar(i) => {
                if i > COLS {
                    Err(format!(
                        "Scalar index {} must be < COLS ({})", 
                        i, COLS))
                } else {
                    for row_id in 0..self.data.len() {
                        res.data[row_id][0] = self.data[row_id][i]
                    }
                    Ok(res)
                }
            }
            TensorIndex::Mask(mask) => {
                for (row_id, (data_row, mask_row)) in zip(self.data, mask.data).enumerate() {
                    let mask_row_index = mask_row.iter().position(|el| *el == 1).ok_or_else(|| "Invalid mask")?;
                    res.data[row_id][0] = data_row[mask_row_index];
                }
                Ok(res)
            }
        }
    }
    pub fn mean(&self) -> T 
    where 
        T: Copy + Sum + ops::Div<Output = T> + FromPrimitive + Default,
    {
        let sum = self.sum();
        let count = T::from_usize(ROWS * COLS).expect("Failed to convert usize to T");
        sum / count
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

impl<T, const ROWS: usize, const COLS: usize> ops::Sub for Tensor<T, ROWS, COLS>
where
    T: ops::Sub<Output = T> + Copy,  // Element type must support addition and be copyable
{
    type Output = Self;  // Result of addition is another Tensor with same dimensions

    fn sub(self, rhs: Self) -> Tensor<T, ROWS, COLS> {
        let mut result = self;  // Copy self (requires T: Copy)
        
        // Element-wise addition
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = result.data[i][j] - rhs.data[i][j];
            }
        }
        
        result
    }
}

impl<T, const ROWS: usize, const COLS: usize> ops::Sub<T> for Tensor<T, ROWS, COLS>
where
    T: ops::Sub<Output = T> + Copy,  // Element type must support addition and be copyable
{
    type Output = Self;  // Result of addition is another Tensor with same dimensions

    fn sub(self, scalar: T) -> Tensor<T, ROWS, COLS> {
        let mut result = self;  // Copy self (requires T: Copy)
        
        // Element-wise addition
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = result.data[i][j] - scalar;
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
    type Output = Self;  // Result of addition is another Tensor with same dimensions

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


impl<T, const ROWS: usize, const COLS: usize> ops::Div<T> for Tensor<T, ROWS, COLS>
where
    T: Default + ops::Div<Output = T> + Copy,  // Element type must support addition and be copyable
{
    type Output = Self;  // Result of addition is another Tensor with same dimensions

    fn div(self, scalar: T) -> Tensor<T, ROWS, COLS> {
        let mut result = self;
        
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = result.data[i][j] / scalar;
            }
        }
                
        result
    }
}

impl<T, const ROWS: usize, const COLS: usize> ops::Div for Tensor<T, ROWS, COLS>
where
    T: Default + ops::Div<Output = T> + Copy,  // Element type must support addition and be copyable
{
    type Output = Self;  // Result of addition is another Tensor with same dimensions

    fn div(self, rhs: Tensor<T, ROWS, COLS>) -> Tensor<T, ROWS, COLS> {

        let mut result = Tensor::<T, ROWS, COLS>::new();
        
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = self.data[i][j] / rhs.data[i][j];
            }
        }
        
        result
    }
}