use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignmentError {
    EmptyMatrix,
    InvalidDataLength { expected: usize, found: usize },
    MatrixTooLarge,
}

impl fmt::Display for AssignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix must have at least one row and one column"),
            Self::InvalidDataLength { expected, found } => write!(f, "Invalid data buffer length: expected {} elements, found {}", expected, found),
            Self::MatrixTooLarge => write!(f, "Matrix dimensions are too large and overflow usize"),
        }
    }
}

impl std::error::Error for AssignmentError {}