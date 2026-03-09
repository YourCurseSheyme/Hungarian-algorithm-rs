use std::ops::{Add, Sub};

/// Trait defining the mathematical operations required for the Hungarian algorithm
///
/// # Float types and NaN
/// Floating-point types(`f32`, `f64`) are intentionally excluded from the default
/// implementations because they do not implement `Ord` trait. The `NaN` (Not a Number)
/// value breaks  the total ordering requirement, which is critical for the algorithm's
/// minimum-finding logic. If you need to use floats, consider using a wrapper like
/// `OrderedFloat` from the `ordered-float` crate and implementing this trait for it.
///
/// # Overflow behavior
/// The Hungarian algorithm primarily performs subtractions (reducing rows/columns) and
/// bounded additions (updating potentials). Mathematically, the internal values never
/// exceed the maximum absolute value present in the initial cost matrix. Therefore,
/// explicit overflow handling (like `checked_add`) is omitted  in the hot path for
/// performance reasons. Note that calculating the `total_cost` at the end *can* overflow
/// if the sum of the assigned weights exceeds the type's capacity.
///
/// # Infinity and `max_value`
/// The `max_value()` is used strictly as an "infinity" sentinel for initializing
/// minimum-search variables (e.g., the `slack` array). It must be handled carefully:
/// the algorithm guarantees it will never add to `max_value()`, which would cause an
/// overflow panic. It is only used for comparisons and replacements.
pub trait Cost: Copy + Ord + Add<Output = Self> + Sub<Output = Self> {
    /// Returns the zero value (additive identity) for the type.
    fn zero() -> Self;

    /// Returns the maximum possible value for the type.
    /// Used as an "infinity" sentinel during minimum searches.
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

impl_cost_for_int!(i32, i64, i128, isize, u32, u64, u128, usize);

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

        assert_eq!(<u64 as Cost>::zero(), 0);
        assert_eq!(<u64 as Cost>::max_value(), u64::MAX);
    }

    #[test]
    fn test_generic_cost_bounds() {
        check_generic_cost::<i32>();
        check_generic_cost::<i64>();
        check_generic_cost::<isize>();
        check_generic_cost::<u32>();
        check_generic_cost::<u64>();
        check_generic_cost::<usize>();
    }
}