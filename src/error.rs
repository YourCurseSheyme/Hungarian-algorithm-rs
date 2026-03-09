use std::fmt;

/// Errors encountered during matrix initialization or solver execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignmentError {
    /// The matrix was initialized with zero rows or zero columns.
    EmptyMatrix,
    /// The provided data buffer length does not match the rows * cols requirement.
    InvalidDataLength { expected: usize, found: usize },
    /// The requested matrix dimensions cause a usize overflow.
    MatrixTooLarge,
}

impl fmt::Display for AssignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix must have at least one row and one column"),
            Self::InvalidDataLength { expected, found } => write!(
                f,
                "Invalid data buffer length: expected {} elements, found {}",
                expected, found
            ),
            Self::MatrixTooLarge => write!(f, "Matrix dimensions are too large and overflow usize"),
        }
    }
}

impl std::error::Error for AssignmentError {}
