//! A high-performance, zero-allocation solver for the Linear Assignment Problem
//! using the Hungarian Algorithm (Kuhn-Munkres) with LAPJV heuristics.

pub(crate) mod algorithm;
pub mod cost;
pub mod error;
pub mod matrix;
pub mod types;

pub use cost::Cost;
pub use error::AssignmentError;
pub use matrix::CostMatrix;
pub use types::{Assignment, Objective};

use algorithm::{HungarianState, UNASSIGNED};

/// Solves the assignment problem for a given cost matrix.
///
/// # Arguments
/// * `matrix` - A reference to the `CostMatrix`.
/// * `objective` - The optimization goal (`Minimize` or `Maximize`).
///
/// # Returns
/// An `Assignment` struct containing the optimal pairs, unmatched entities,
/// and the total computed cost.
pub fn solve_assignment<C: Cost>(
    matrix: &CostMatrix<C>,
    objective: Objective,
) -> Result<Assignment<C>, AssignmentError> {
    if matrix.is_empty() {
        return Err(AssignmentError::EmptyMatrix);
    }

    let mut state = HungarianState::new(matrix, objective);
    state.solve();

    let mut pairs = Vec::with_capacity(matrix.rows().min(matrix.cols()));
    let mut total_cost = C::zero();

    let mut matched_rows = vec![false; matrix.rows()];
    let mut matched_cols = vec![false; matrix.cols()];

    for i in 0..state.n {
        let j = state.xy[i];
        if j != UNASSIGNED {
            if i < matrix.rows() && j < matrix.cols() {
                pairs.push((i, j));
                total_cost = total_cost + matrix[(i, j)];
                matched_rows[i] = true;
                matched_cols[j] = true;
            }
        }
    }

    let unmatched_rows: Vec<usize> = matched_rows
        .into_iter()
        .enumerate()
        .filter_map(|(i, matched)| if !matched { Some(i) } else { None })
        .collect();

    let unmatched_cols: Vec<usize> = matched_cols
        .into_iter()
        .enumerate()
        .filter_map(|(j, matched)| if !matched { Some(j) } else { None })
        .collect();

    Ok(Assignment::new(
        pairs,
        unmatched_rows,
        unmatched_cols,
        total_cost,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_api_minimize() {
        let matrix = CostMatrix::new(3, 3, vec![8, 4, 7, 5, 2, 3, 9, 4, 8]).unwrap();

        let assignment = solve_assignment(&matrix, Objective::Minimize).unwrap();

        assert_eq!(assignment.total_cost(), 15);
        assert_eq!(assignment.pairs().len(), 3);
        assert!(assignment.unmatched_rows().is_empty());
    }

    #[test]
    fn test_public_api_maximize() {
        let matrix = CostMatrix::new(3, 3, vec![8, 4, 7, 5, 2, 3, 9, 4, 8]).unwrap();

        let assignment = solve_assignment(&matrix, Objective::Maximize).unwrap();

        assert_eq!(assignment.total_cost(), 18);
    }

    #[test]
    fn test_rectangular_unmatched() {
        let matrix = CostMatrix::new(2, 3, vec![10, 20, 30, 40, 50, 60]).unwrap();

        let assignment = solve_assignment(&matrix, Objective::Maximize).unwrap();

        assert_eq!(assignment.total_cost(), 80);
        assert_eq!(assignment.pairs().len(), 2);
        assert_eq!(assignment.unmatched_rows().len(), 0);
        assert_eq!(assignment.unmatched_cols(), vec![0]);
    }
}
