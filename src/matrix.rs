use crate::error::AssignmentError;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> CostMatrix<T> {
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, AssignmentError> {
        if rows == 0 || cols == 0 {
            return Err(AssignmentError::EmptyMatrix);
        }

        let expected_len = rows.checked_mul(cols).ok_or(AssignmentError::MatrixTooLarge)?;

        if data.len() != expected_len {
            return Err(AssignmentError::InvalidDataLength {
                expected: expected_len,
                found: data.len(),
            });
        }

        Ok(Self { rows, cols, data })
    }

    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Result<Self, AssignmentError>
    where
        F: FnMut(usize, usize) -> T,
    {
        if rows == 0 || cols == 0 {
            return Err(AssignmentError::EmptyMatrix);
        }

        let capacity = rows.checked_mul(cols).ok_or(AssignmentError::MatrixTooLarge)?;
        let mut data = Vec::with_capacity(capacity);

        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }

        Ok(Self { rows, cols, data })
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn into_inner(self) -> Vec<T> {
        self.data
    }

    #[inline(always)]
    fn linear_index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[self.linear_index(row, col)])
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            let idx = self.linear_index(row, col);
            Some(&mut self.data[idx])
        } else {
            None
        }
    }
}

impl<T> Index<(usize, usize)> for CostMatrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        debug_assert!(row < self.rows, "Row index out of bounds");
        debug_assert!(col < self.cols, "Column index out of bounds");

        &self.data[self.linear_index(row, col)]
    }
}

impl<T> IndexMut<(usize, usize)> for CostMatrix<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        debug_assert!(row < self.rows, "Row index out of bounds");
        debug_assert!(col < self.cols, "Column index out of bounds");

        let idx = self.linear_index(row, col);
        &mut self.data[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation_success() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let matrix = CostMatrix::new(2, 3, data).unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
    }

    #[test]
    fn test_matrix_from_fn() {
        let matrix = CostMatrix::from_fn(2, 2, |r, c| r + c).unwrap();
        assert_eq!(matrix[(0, 0)], 0);
        assert_eq!(matrix[(0, 1)], 1);
        assert_eq!(matrix[(1, 0)], 1);
        assert_eq!(matrix[(1, 1)], 2);
    }

    #[test]
    fn test_matrix_empty_error() {
        let err = CostMatrix::<i32>::new(0, 5, vec![]).unwrap_err();
        assert_eq!(err, AssignmentError::EmptyMatrix);
    }

    #[test]
    fn test_matrix_invalid_data_length() {
        let data = vec![1, 2, 3];
        let err = CostMatrix::new(2, 2, data).unwrap_err();
        assert_eq!(err, AssignmentError::InvalidDataLength { expected: 4, found: 3 });
    }

    #[test]
    fn test_matrix_too_large() {
        let err = CostMatrix::<i32>::new(usize::MAX, 2, Vec::new()).unwrap_err();
        assert_eq!(err, AssignmentError::MatrixTooLarge);
    }

    #[test]
    fn test_matrix_safe_get() {
        let mut matrix = CostMatrix::new(2, 2, vec![10, 20, 30, 40]).unwrap();
        assert_eq!(matrix.get(0, 1), Some(&20));
        assert_eq!(matrix.get(2, 0), None);

        if let Some(val) = matrix.get_mut(1, 1) {
            *val = 99;
        }
        assert_eq!(matrix.get(1, 1), Some(&99));
    }

    #[test]
    fn test_matrix_indexing() {
        let mut matrix = CostMatrix::new(2, 2, vec![10, 20, 30, 40]).unwrap();
        assert_eq!(matrix[(0, 1)], 20);
        assert_eq!(matrix[(1, 0)], 30);
        matrix[(1, 1)] = 99;
        assert_eq!(matrix[(1, 1)], 99);
    }

    #[test]
    #[should_panic(expected = "Row index out of bounds")]
    fn test_matrix_out_of_bounds_row() {
        let matrix = CostMatrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let _ = matrix[(2, 0)];
    }

    #[test]
    #[should_panic(expected = "Column index out of bounds")]
    fn test_matrix_out_of_bounds_col() {
        let matrix = CostMatrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let _ = matrix[(0, 2)];
    }

    #[test]
    fn test_matrix_clone() {
        let m1 = CostMatrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let m2 = m1.clone();
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_matrix_shape_and_into_inner() {
        let matrix = CostMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(matrix.shape(), (2, 3));

        let vec = matrix.into_inner();
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6]);
    }
}