use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

define_ty!(U32x8, u32, u32, u32, u32, u32, u32, u32, u32);
impl_minimal!(U32x8, u32, 8, x0, x1, x2, x3, x4, x5, x6, x7);

impl U32x8 {
    /// Create a new `U32x8` from a slice.
    ///
    /// # Panics
    ///
    /// Will panic if the slice is not at least 4 elements long.
    pub fn from_slice(slice: &[u32]) -> Self {
        debug_assert!(slice.len() >= Self::lanes());
        Self(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        )
    }

    pub fn horizontal_add(self) -> u32 {
        self.0 + self.1 + self.2 + self.3 + self.4 + self.5 + self.6 + self.7
    }
}

impl_op8!(Mul, mul, U32x8, *);
impl_op8!(assn MulAssign, mul_assign, U32x8, *=);
impl_op8!(Div, div, U32x8, /);
impl_op8!(assn DivAssign, div_assign, U32x8, /=);
impl_op8!(Add, add, U32x8, +);
impl_op8!(assn AddAssign, add_assign, U32x8, +=);
impl_op8!(Sub, sub, U32x8, -);
impl_op8!(assn SubAssign, sub_assign, U32x8, -=);

impl_distances_int!(U32x8, u32, f32);
