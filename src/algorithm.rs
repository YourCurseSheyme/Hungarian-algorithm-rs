use crate::types::Objective;
use crate::cost::Cost;
use crate::matrix::CostMatrix;

/// Marker indicating the absence of an assignment for a row or column.
/// Using a constant instead of `Option<usize>` keeps the array element size
/// at 8 bytes (on 64-bit systems), significantly improving CPU cache locality.
pub(crate) const UNASSIGNED: usize = usize::MAX;

/// Internal state for the Hungarian algorithm.
///
/// # Non-negative weights requirement
/// The algorithm requires all weights in the cost matrix to be non-negative ('>= 0').
/// This is mathematically necessary because we use `0` as the cost for virtual edges
/// (padding) when dealing with rectangular matrices. If negative weights were allowed,
/// the algorithm might prefer real assignments over virtual ones, breaking the logic
/// of finding te optimal matching for asymmetric sets.
///
/// # Memory management
/// This struct holds all necessary buffers. Memory is allocated exactly once
/// during initialization, guaranteeing zero allocations in the hot O(N^3) loop.
pub(crate) struct HungarianState<'a, C: Cost> {
    matrix: &'a CostMatrix<C>,
    rows: usize,
    cols: usize,
    pub(crate) n: usize,
    objective: Objective,
    max_weights: C,

    u: Vec<C>,
    v: Vec<C>,

    pub(crate) xy: Vec<usize>,
    pub(crate) yx: Vec<usize>,

    slack: Vec<C>,
    slackx: Vec<usize>,
    prev: Vec<usize>,

    visited_left: Vec<usize>,
    visited_right: Vec<usize>,
    generation: usize,
}

impl <'a, C: Cost> HungarianState<'a, C> {
    pub(crate) fn new(matrix: &'a CostMatrix<C>, objective: Objective) -> Self {
        let (rows, cols) = matrix.shape();
        let n = rows.max(cols);

        let mut max_weights = C::zero();
        if objective == Objective::Maximize && !matrix.is_empty() {
            max_weights = matrix.data()[0];
            for &val in matrix.data() {
                if val > max_weights {
                    max_weights = val;
                }
            }
        }

        Self {
            matrix,
            rows,
            cols,
            n,
            objective,
            max_weights,
            u: vec![C::zero(); n],
            v: vec![C::zero(); n],
            xy: vec![UNASSIGNED; n],
            yx: vec![UNASSIGNED; n],
            slack: vec![C::max_value(); n],
            slackx: vec![0; n],
            prev: vec![UNASSIGNED; n],
            visited_left: vec![0; n],
            visited_right: vec![0; n],
            generation: 0,
        }
    }

    #[inline]
    pub fn cost(&self, row: usize, col: usize) -> C {
        if row < self.rows && col < self.cols {
            let val = self.matrix[(row, col)];
            match self.objective {
                Objective::Minimize => val,
                Objective::Maximize => self.max_weights - val,
            }
        } else {
            C::zero()
        }
    }

    #[inline]
    fn clear_for_step(&mut self) {
        self.generation += 1;
        self.slack.fill(C::max_value());
        self.prev.fill(UNASSIGNED);
    }

    pub(crate) fn solve(&mut self) {
        if self.n == 0 {
            return;
        }

        self.initial_reduction();
        self.compute_initial_matching();

        for i in 0..self.n {
            if self.xy[i] == UNASSIGNED {
                self.augment(i);
            }
        }
    }

    fn initial_reduction(&mut self) {
        if self.cols >= self.rows {
            for i in 0..self.rows {
                let mut min_val = C::max_value();
                for j in 0..self.cols {
                    let c = self.cost(i, j);
                    if c < min_val {
                        min_val = c;
                    }
                }
                self.u[i] = min_val;
            }
        }

        if self.rows >= self.cols {
            for j in 0..self.cols {
                let mut min_val = C::max_value();
                for i in 0..self.rows {
                    let reduced_cost = self.cost(i, j) - self.u[i];
                    if reduced_cost < min_val {
                        min_val = reduced_cost;
                    }
                }
                self.v[j] = min_val;
            }
        }
    }

    fn compute_initial_matching(&mut self) {
        for i in 0..self.n {
            for j in 0..self.n {
                if self.xy[i] == UNASSIGNED && self.yx[j] == UNASSIGNED {
                    if self.cost(i, j) - self.u[i] - self.v[j] == C::zero() {
                        self.xy[i] = j;
                        self.yx[j] = i;
                        break;
                    }
                }
            }
        }
    }

    fn augment(&mut self, root: usize) {
        self.clear_for_step();

        let mut current_row = root;
        self.visited_left[current_row] = self.generation;

        for j in 0..self.n {
            self.slack[j] = self.cost(current_row, j) - self.u[current_row] - self.v[j];
            self.slackx[j] = current_row;
        }

        let end_col;

        loop {
            let mut min_slack = C::max_value();
            let mut j0 = UNASSIGNED;

            for j in 0..self.n {
                if self.visited_right[j] != self.generation && self.slack[j] < min_slack {
                    min_slack = self.slack[j];
                    j0 = j;
                }
            }

            debug_assert!(min_slack < C::max_value(), "No augmenting path possible");

            if min_slack > C::zero() {
                for i in 0..self.n {
                    if self.visited_left[i] == self.generation {
                        self.u[i] = self.u[i] + min_slack;
                    }
                }
                for j in 0..self.n {
                    if self.visited_right[j] == self.generation {
                        self.v[j] = self.v[j] - min_slack;
                    } else {
                        self.slack[j] = self.slack[j] - min_slack;
                    }
                }
            }

            self.visited_right[j0] = self.generation;
            self.prev[j0] = self.slackx[j0];

            if self.yx[j0] == UNASSIGNED {
                end_col = j0;
                break;
            }

            current_row = self.yx[j0];
            self.visited_left[current_row] = self.generation;

            for j in 0..self.n {
                if self.visited_right[j] != self.generation {
                    let reduced_cost = self.cost(current_row, j) - self.u[current_row] - self.v[j];
                    if reduced_cost < self.slack[j] {
                        self.slack[j] = reduced_cost;
                        self.slackx[j] = current_row;
                    }
                }
            }
        }

        let mut j = end_col;
        while j != UNASSIGNED {
            let i = self.prev[j];
            let next_j = self.xy[i];
            self.yx[j] = i;
            self.xy[i] = j;
            j = next_j;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::CostMatrix;

    #[test]
    fn test_sentinel_matching() {
        assert_eq!(UNASSIGNED, usize::MAX);
    }

    #[test]
    fn test_cached_dimensions() {
        let matrix = CostMatrix::new(2, 5, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();
        let state = HungarianState::new(&matrix, Objective::Minimize);

        assert_eq!(state.rows, 2);
        assert_eq!(state.cols, 5);
        assert_eq!(state.n, 5);
    }

    #[test]
    fn test_state_initialization() {
        let matrix = CostMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let state = HungarianState::new(&matrix, Objective::Minimize);

        assert_eq!(state.u.len(), 3);
        assert_eq!(state.xy.len(), 3);
        assert_eq!(state.xy[0], UNASSIGNED);
        assert_eq!(state.slack[0], i32::MAX);
        assert_eq!(state.prev[0], UNASSIGNED);
    }

    #[test]
    fn test_virtual_padding_cost() {
        let matrix = CostMatrix::new(2, 3, vec![10, 20, 30, 40, 50, 60]).unwrap();
        let state = HungarianState::new(&matrix, Objective::Minimize);

        assert_eq!(state.cost(0, 1), 20);
        assert_eq!(state.cost(1, 2), 60);

        assert_eq!(state.cost(2, 0), 0);
        assert_eq!(state.cost(2, 2), 0);

        assert_eq!(state.cost(0, 3), 0);
    }

    #[test]
    fn test_clear_for_step() {
        let matrix = CostMatrix::new(1, 1, vec![1]).unwrap();
        let mut state = HungarianState::new(&matrix, Objective::Minimize);

        state.visited_left[0] = 1;
        state.visited_right[0] = 1;
        state.slack[0] = 42;
        state.prev[0] = 0;

        state.clear_for_step();

        assert_eq!(state.visited_left[0], 1);
        assert_eq!(state.visited_right[0], 1);
        assert_eq!(state.slack[0], i32::MAX);
        assert_eq!(state.prev[0], UNASSIGNED);
    }

    #[test]
    fn test_solve_3x3() {
        let matrix = CostMatrix::new(3, 3, vec![
            8, 4, 7,
            5, 2, 3,
            9, 4, 8
        ]).unwrap();

        let mut state = HungarianState::new(&matrix, Objective::Minimize);
        state.solve();

        assert_eq!(state.xy[0], 0);
        assert_eq!(state.xy[1], 2);
        assert_eq!(state.xy[2], 1);
    }

    #[test]
    fn test_solve_rectangular_2x3() {
        let matrix = CostMatrix::new(2, 3, vec![
            10, 20, 30,
            40, 50, 60,
        ]).unwrap();

        let mut state = HungarianState::new(&matrix, Objective::Minimize);
        state.solve();

        assert_eq!(state.xy[0], 0);
        assert_eq!(state.xy[1], 1);
        assert_eq!(state.xy[2], 2);
    }

    #[test]
    fn test_solve_all_zeros() {
        let matrix = CostMatrix::new(3, 3, vec![0; 9]).unwrap();

        let mut state = HungarianState::new(&matrix, Objective::Minimize);
        state.solve();

        assert!(state.xy.iter().all(|&j| j != UNASSIGNED));
        assert!(state.yx.iter().all(|&i| i != UNASSIGNED));
    }
}