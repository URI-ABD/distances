use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

define_ty!(U32x4, u32, u32, u32, u32);
impl_minimal!(U32x4, u32, 4, x0, x1, x2, x3);

impl U32x4 {
    /// Create a new `U32x4` from a slice.
    ///
    /// # Panics
    ///
    /// Will panic if the slice is not at least 4 elements long.
    pub fn from_slice(slice: &[u32]) -> Self {
        debug_assert!(slice.len() >= Self::lanes());
        Self(slice[0], slice[1], slice[2], slice[3])
    }

    pub const fn horizontal_add(self) -> u32 {
        self.0 + self.1 + self.2 + self.3
    }
}

impl_op4!(Mul, mul, U32x4, *);
impl_op4!(assn MulAssign, mul_assign, U32x4, *=);
impl_op4!(Div, div, U32x4, /);
impl_op4!(assn DivAssign, div_assign, U32x4, /=);
impl_op4!(Add, add, U32x4, +);
impl_op4!(assn AddAssign, add_assign, U32x4, +=);
impl_op4!(Sub, sub, U32x4, -);
impl_op4!(assn SubAssign, sub_assign, U32x4, -=);

impl_distances_uint!(U32x4, u32, f32);
