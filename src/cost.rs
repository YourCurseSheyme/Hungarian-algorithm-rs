use std::ops::{Add, Sub};

/// Mathematical operations required for the Hungarian algorithm.
///
/// # Type Constraints
/// Requires Ord to guarantee total ordering. Floating-point types (f32, f64)
/// are intentionally excluded due to NaN values breaking the minimum-search logic.
/// Use a wrapper like OrderedFloat if fractional costs are strictly required.
///
/// # Overflow Behavior
/// The algorithm primarily performs bounded subtractions and additions. Internal
/// values never exceed the maximum absolute value present in the initial matrix.
/// Explicit overflow handling is omitted in the hot path for maximum performance.
pub trait Cost: Copy + Ord + Add<Output = Self> + Sub<Output = Self> {
    /// Returns the additive identity (zero) for the type.
    fn zero() -> Self;

    /// Returns the maximum possible value for the type.
    /// Used strictly as an "infinity" sentinel during minimum searches.
    fn max_value() -> Self;
}

macro_rules! impl_cost_for_int {
    ($($t:ty),*) => {
        $(
        impl Cost for $t {
            fn zero() -> Self {
                0
            }

            fn max_value() -> Self {
                <$t>::MAX
            }
        }
        )*
    };
}

// Restricts implementation to signed integers.
// Dual variables (potentials) in the classic algorithm can become negative.
impl_cost_for_int!(i32, i64, i128, isize);

#[cfg(test)]
mod tests {
    use super::*;

    fn check_generic_cost<T: Cost>() {
        let zero = T::zero();
        let max = T::max_value();
        assert!(max > zero);
    }

    #[test]
    fn test_cost_implementation() {
        assert_eq!(<i32 as Cost>::zero(), 0);
        assert_eq!(<i32 as Cost>::max_value(), i32::MAX);
    }

    #[test]
    fn test_generic_cost_bounds() {
        check_generic_cost::<i32>();
        check_generic_cost::<i64>();
        check_generic_cost::<isize>();
    }
}
