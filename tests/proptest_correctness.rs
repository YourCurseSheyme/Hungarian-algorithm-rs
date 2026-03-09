use proptest::prelude::*;
use assignment_solver::{CostMatrix, Objective, solve_assignment};
use std::collections::HashSet;

prop_compose! {
    fn matrix_strategy(max_rows: usize, max_cols: usize)(
        rows in 1..=max_rows,
        cols in 1..=max_cols,
    )(
        rows in Just(rows),
        cols in Just(cols),
        data in prop::collection::vec(-1000i32..1000i32, rows * cols)
    ) -> CostMatrix<i32> {
        CostMatrix::new(rows, cols, data).unwrap()
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    #[test]
    fn test_assignment_invariants(matrix in matrix_strategy(50, 50)) {
        let rows = matrix.rows();
        let cols = matrix.cols();
        let min_dim = rows.min(cols);

        let assignment = solve_assignment(&matrix, Objective::Minimize).unwrap();

        prop_assert_eq!(assignment.pairs().len(), min_dim);

        let mut seen_rows = HashSet::new();
        let mut seen_cols = HashSet::new();
        let mut calculated_cost = 0;

        for &(r, c) in assignment.pairs() {
            prop_assert!(r < rows);
            prop_assert!(c < cols);

            prop_assert!(seen_rows.insert(r), "Row {} assigned multiple times", r);
            prop_assert!(seen_cols.insert(c), "Column {} assigned multiple times", c);

            calculated_cost += matrix[(r, c)];
        }

        prop_assert_eq!(assignment.total_cost(), calculated_cost);

        prop_assert_eq!(assignment.unmatched_rows().len(), rows - min_dim);
        prop_assert_eq!(assignment.unmatched_cols().len(), cols - min_dim);
    }
}