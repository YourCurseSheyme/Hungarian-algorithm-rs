#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment<T> {
    pairs: Vec<(usize, usize)>,
    unmatched_rows: Vec<usize>,
    unmatched_cols: Vec<usize>,
    total_cost: T,
}

impl<T> Assignment<T> {
    pub(crate) fn new(
        pairs: Vec<(usize, usize)>,
        unmatched_rows: Vec<usize>,
        unmatched_cols: Vec<usize>,
        total_cost: T
    ) -> Self {
        Self {
            pairs,
            unmatched_rows,
            unmatched_cols,
            total_cost,
        }
    }
    
    #[inline]
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }
    
    #[inline]
    pub fn unmatched_rows(&self) -> &[usize] {
        &self.unmatched_rows
    }
    
    #[inline]
    pub fn unmatched_cols(&self) -> &[usize] {
        &self.unmatched_cols
    }
}

impl<T: Copy> Assignment<T> {
    #[inline]
    pub fn total_cost(&self) -> T {
        self.total_cost
    }
}