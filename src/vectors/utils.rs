//! Utility functions for vector based distance calculations.

use crate::Number;

/// An iterator over the absolute differences between the corresponding elements
/// of two vectors.
///
/// If the two vectors have differing or zero dimensionality, this function will
/// panic in debug mode, and may silently give incorrect values in release mode.
pub fn abs_diff_iter<'a, T: Number>(x: &'a [T], y: &'a [T]) -> impl Iterator<Item = T> + 'a {
    debug_assert_eq!(x.len(), y.len());
    debug_assert!(!x.is_empty());

    x.iter().zip(y.iter()).map(|(a, &b)| a.abs_diff(b))
}

// /// An iterator over the differences between the corresponding elements of two
// /// slices. The elements of the second slice are subtracted from those of the
// /// first. It is the user's responsibility to ensure that there is no overflow.
// pub fn diff_iter<'a, T: Number>(x: &'a [T], y: &'a [T]) -> impl Iterator<Item = T> + 'a {
//     x.iter().zip(y.iter()).map(|(&a, &b)| a - b)
// }
