//! Number variants for floats, integers, and unsigned integers.

use core::hash::Hash;

use crate::Number;

/// Sub-trait of `Number` for all integer types.
pub trait Int: Number + Hash + Eq + Ord {}

/// Macro to implement `IntNumber` for all integer types.
macro_rules! impl_int {
    ($($ty:ty),*) => {
        $(
            impl Int for $ty {}
        )*
    }
}

impl_int!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize);

/// Sub-trait of `Number` for all signed integer types.
pub trait IInt: Number + Hash + Eq + Ord {}

/// Macro to implement `IIntNumber` for all signed integer types.
macro_rules! impl_iint {
    ($($ty:ty),*) => {
        $(
            impl IInt for $ty {}
        )*
    }
}

impl_iint!(i8, i16, i32, i64, i128, isize);

/// Sub-trait of `Number` for all unsigned integer types.
pub trait UInt: Number + Hash + Eq + Ord {
    /// Returns the number as a `i64`.
    fn as_i64(self) -> i64;

    /// Returns the number as a `u64`.
    fn as_u64(self) -> u64;
}

/// Macro to implement `UIntNumber` for all unsigned integer types.
macro_rules! impl_uint {
    ($($ty:ty),*) => {
        $(
            impl UInt for $ty {
                fn as_i64(self) -> i64 {
                    self as i64
                }

                fn as_u64(self) -> u64 {
                    self as u64
                }
            }
        )*
    }
}

impl_uint!(u8, u16, u32, u64, u128, usize);

/// Sub-trait of `Number` for all floating point types.
pub trait Float: Number {
    /// Returns the square root of a `Float`.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Returns the cube root of a `Float`.
    #[must_use]
    fn cbrt(self) -> Self;

    /// Returns the fourth root of a `Float`.
    #[must_use]
    fn fort(self) -> Self {
        self.sqrt().sqrt()
    }

    /// The square-root of 2.
    #[must_use]
    fn sqrt_2() -> Self;

    /// Returns the machine epsilon for a `Float`.
    fn epsilon() -> Self;

    /// Whether the Float is positive
    fn is_pos(self) -> bool;

    /// Returns the inverse square root of a `Float`, i.e. `1.0 / self.sqrt()`.
    #[must_use]
    fn inv_sqrt(self) -> Self {
        Self::one() / self.sqrt()
    }

    /// Returns `self` raised to the power of `exp`.
    #[must_use]
    fn powf(self, exp: Self) -> Self;

    /// Error function
    ///
    /// Calculates an approximation to the “error function”, which estimates
    /// the probability that an observation will fall within x standard
    /// deviations of the mean (assuming a normal distribution).
    #[must_use]
    fn erf(self) -> Self;

    /// Efficient implementation of Sigmoid function, \\( S(x) = \frac{1}{1 + e^{-x}} \\), see [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
    #[must_use]
    fn sigmoid(self) -> Self;

    /// Returns the base 2 logarithm.
    #[must_use]
    fn log2(self) -> Self;
}

impl Float for f32 {
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    fn cbrt(self) -> Self {
        Self::cbrt(self)
    }

    fn sqrt_2() -> Self {
        core::f32::consts::SQRT_2
    }

    fn epsilon() -> Self {
        Self::EPSILON
    }

    fn is_pos(self) -> bool {
        self.is_sign_positive()
    }

    fn powf(self, exp: Self) -> Self {
        Self::powf(self, exp)
    }

    fn erf(self) -> Self {
        libm::erff(self)
    }

    fn sigmoid(self) -> Self {
        if self < -40. {
            0.
        } else if self > 40. {
            1.
        } else {
            1. / (1. + Self::exp(-self))
        }
    }

    fn log2(self) -> Self {
        Self::log2(self)
    }
}

impl Float for f64 {
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    fn cbrt(self) -> Self {
        Self::cbrt(self)
    }

    fn sqrt_2() -> Self {
        core::f64::consts::SQRT_2
    }

    fn epsilon() -> Self {
        Self::EPSILON
    }

    fn is_pos(self) -> bool {
        self.is_sign_positive()
    }

    fn powf(self, exp: Self) -> Self {
        Self::powf(self, exp)
    }

    fn erf(self) -> Self {
        libm::erf(self)
    }

    fn sigmoid(self) -> Self {
        if self < -40. {
            0.
        } else if self > 40. {
            1.
        } else {
            1. / (1. + Self::exp(-self))
        }
    }

    fn log2(self) -> Self {
        Self::log2(self)
    }
}
