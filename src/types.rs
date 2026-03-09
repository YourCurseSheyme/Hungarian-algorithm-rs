/// Optimization objective for the assignment solver.
///
/// Defines whether the algorithm should minimize or maximize the total cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Finds the assignment that results in the lowest possible total cost.
    Minimize,
    /// Finds the assignment that results in the highest possible total cost.
    Maximize,
}

/// Result of the assignment algorithm.
///
/// Contains the optimal bipartite matching pairs, the total computed cost,
/// and any unmatched entities (in the case of rectangular matrices).
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
        total_cost: T,
    ) -> Self {
        Self {
            pairs,
            unmatched_rows,
            unmatched_cols,
            total_cost,
        }
    }

    /// Returns the optimal assignment pairs as `(row_index, column_index)`.
    #[inline]
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Returns the indices of rows that were left unassigned.
    /// Only populated if `rows > cols`.
    #[inline]
    pub fn unmatched_rows(&self) -> &[usize] {
        &self.unmatched_rows
    }

    /// Returns the indices of columns that were left unassigned.
    /// Only populated if `cols > rows`.
    #[inline]
    pub fn unmatched_cols(&self) -> &[usize] {
        &self.unmatched_cols
    }
}

impl<T: Copy> Assignment<T> {
    /// Returns the total computed cost of the optimal assignment.
    #[inline]
    pub fn total_cost(&self) -> T {
        self.total_cost
    }
}
