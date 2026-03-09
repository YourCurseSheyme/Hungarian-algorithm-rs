use crate::cost::Cost;
use crate::matrix::CostMatrix;
use crate::types::Objective;

/// Sentinel value representing an unassigned row or column.
pub(crate) const UNASSIGNED: usize = usize::MAX;

/// Internal execution state of the Hungarian algorithm.
///
/// Designed for zero-allocation in the hot path. All required buffers are
/// allocated exactly once during initialization.
pub(crate) struct HungarianState<'a, C: Cost> {
    /// Flattened slice of the cost matrix for direct, cache-friendly access.
    matrix_data: &'a [C],
    rows: usize,
    cols: usize,
    /// Virtual square matrix dimension: `max(rows, cols)`.
    pub(crate) n: usize,
    objective: Objective,
    /// Maximum weight in the matrix, used for on-the-fly maximization inversion.
    max_weight: C,

    /// Row potentials (Dual variables).
    u: Vec<C>,
    /// Column potentials (Dual variables).
    v: Vec<C>,

    /// Row-to-column assignments. `xy[i] = j` means row `i` is assigned to column `j`.
    pub(crate) xy: Vec<usize>,
    /// Column-to-row assignments. `yx[j] = i` means column `j` is assigned to row `i`.
    pub(crate) yx: Vec<usize>,

    /// Minimum reduced cost to reach column `j` from the current alternating tree.
    slack: Vec<C>,
    /// Row index that provides the minimum slack for column `j`.
    slackx: Vec<usize>,
    /// Ancestor array used to reconstruct the augmenting path.
    prev: Vec<usize>,

    /// Visited state for rows. Uses a generation counter for O(1) resets.
    visited_left: Vec<usize>,
    /// Visited state for columns. Uses a generation counter for O(1) resets.
    visited_right: Vec<usize>,
    /// Current search generation. Incremented per augmenting path search.
    generation: usize,
}

impl<'a, C: Cost> HungarianState<'a, C> {
    pub(crate) fn new(matrix: &'a CostMatrix<C>, objective: Objective) -> Self {
        let (rows, cols) = matrix.shape();
        let n = rows.max(cols);

        let mut max_weight = C::zero();
        if objective == Objective::Maximize && !matrix.is_empty() {
            max_weight = matrix.data()[0];
            for &val in matrix.data() {
                if val > max_weight {
                    max_weight = val;
                }
            }
        }

        Self {
            matrix_data: matrix.data(),
            rows,
            cols,
            n,
            objective,
            max_weight,
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

    /// Retrieves the assignment cost for a given row and column.
    ///
    /// Implements virtual padding: returns `0` for out-of-bound indices,
    /// allowing rectangular matrices to be processed as square matrices.
    #[inline]
    pub fn cost(&self, row: usize, col: usize) -> C {
        if row < self.rows && col < self.cols {
            let val = self.matrix_data[row * self.cols + col];
            match self.objective {
                Objective::Minimize => val,
                Objective::Maximize => self.max_weight - val,
            }
        } else {
            C::zero()
        }
    }

    /// Prepares buffers for the next augmenting path search.
    /// Achieves O(1) visited state clearance via generation increment.
    #[inline]
    fn clear_for_step(&mut self) {
        self.generation += 1;
        self.slack.fill(C::max_value());
        self.prev.fill(UNASSIGNED);
    }

    /// Primary orchestration method for the algorithm.
    pub(crate) fn solve(&mut self) {
        if self.n == 0 {
            return;
        }

        self.lapjv_initialization();
        self.compute_initial_matching();

        for i in 0..self.n {
            if self.xy[i] == UNASSIGNED {
                self.augment(i);
            }
        }
    }

    /// Jonker-Volgenant (LAPJV) Initialization Heuristic.
    ///
    /// Replaces standard row/column reduction. Utilizes "Reduction Transfer" to
    /// establish 80-95% of optimal assignments before the O(n³) augmentation phase.
    fn lapjv_initialization(&mut self) {
        // Phase 1: Column Reduction
        for j in 0..self.n {
            let mut min_val = C::max_value();
            for i in 0..self.n {
                let c = self.cost(i, j);
                if c < min_val {
                    min_val = c;
                }
            }
            self.v[j] = min_val;
        }

        // Phase 2: Reduction Transfer
        for i in 0..self.n {
            let mut min1 = C::max_value();
            let mut min2 = C::max_value();
            let mut j1 = UNASSIGNED;

            for j in 0..self.n {
                let reduced_cost = self.cost(i, j) - self.v[j];
                if reduced_cost < min1 {
                    min2 = min1;
                    min1 = reduced_cost;
                    j1 = j;
                } else if reduced_cost < min2 {
                    min2 = reduced_cost;
                }
            }

            if self.n > 1 {
                self.u[i] = min2;
                if min1 < min2 {
                    let delta = min2 - min1;
                    self.v[j1] = self.v[j1] - delta;
                }
            } else {
                self.u[i] = min1;
            }
        }
    }

    /// Greedy Initial Matching.
    /// Scans the matrix to claim all available edges with zero reduced cost.
    fn compute_initial_matching(&mut self) {
        for i in 0..self.n {
            let u_i = self.u[i];
            for j in 0..self.n {
                if self.xy[i] == UNASSIGNED && self.yx[j] == UNASSIGNED {
                    if self.cost(i, j) - u_i - self.v[j] == C::zero() {
                        self.xy[i] = j;
                        self.yx[j] = i;
                        break;
                    }
                }
            }
        }
    }

    /// Augmenting Path Search.
    /// Finds an augmenting path for the unassigned `root` row and updates dual variables.
    fn augment(&mut self, root: usize) {
        self.clear_for_step();

        let mut current_row = root;
        self.visited_left[current_row] = self.generation;

        let mut u_i = self.u[current_row];
        for j in 0..self.n {
            self.slack[j] = self.cost(current_row, j) - u_i - self.v[j];
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

            // Dual Update: Maintains the invariant C(i,j) - u[i] - v[j] >= 0
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
            u_i = self.u[current_row];

            for j in 0..self.n {
                if self.visited_right[j] != self.generation {
                    let reduced_cost = self.cost(current_row, j) - u_i - self.v[j];
                    if reduced_cost < self.slack[j] {
                        self.slack[j] = reduced_cost;
                        self.slackx[j] = current_row;
                    }
                }
            }
        }

        // Augmentation: Flip edges along the found path to increase matching size by 1.
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
        let matrix = CostMatrix::new(3, 3, vec![8, 4, 7, 5, 2, 3, 9, 4, 8]).unwrap();

        let mut state = HungarianState::new(&matrix, Objective::Minimize);
        state.solve();

        assert_eq!(state.xy[0], 0);
        assert_eq!(state.xy[1], 2);
        assert_eq!(state.xy[2], 1);
    }

    #[test]
    fn test_solve_rectangular_2x3() {
        let matrix = CostMatrix::new(2, 3, vec![10, 20, 30, 40, 50, 60]).unwrap();

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

    #[test]
    fn test_solve_all_equal() {
        let matrix = CostMatrix::new(4, 4, vec![42; 16]).unwrap();

        let mut state = HungarianState::new(&matrix, Objective::Minimize);
        state.solve();

        assert!(state.xy.iter().all(|&j| j != UNASSIGNED));
        assert!(state.yx.iter().all(|&i| i != UNASSIGNED));

        let mut total_cost = 0;
        for i in 0..4 {
            total_cost += state.cost(i, state.xy[i]);
        }
        assert_eq!(total_cost, 168);
    }
}
