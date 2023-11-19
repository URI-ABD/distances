use std::ops::{Add, Mul};

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug)]
/// A cartesian form complex number over some type T (implemented here as f32 and f64).
/// Of the form a + bi
pub struct CartComplex<T>(T, T);

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug)]
/// A polar form complex number over some type T (implemented here as f32 and f64).
/// Of the form r(cos(theta) + i*sin(theta))
pub struct PolarComplex<T>(T, T);

/// Implements the logic and functionality for cartesian complex numbers
macro_rules! impl_cart_complex {
    ($t:ty) => {
        impl Add for CartComplex<$t> {
            type Output = CartComplex<$t>;

            fn add(self, rhs: Self) -> Self::Output {
                CartComplex(self.0 + rhs.0, self.1 + rhs.1)
            }
        }

        impl Mul for CartComplex<$t> {
            type Output = CartComplex<$t>;

            fn mul(self, rhs: Self) -> Self::Output {
                let (CartComplex(a, b), CartComplex(c, d)) = (self, rhs);
                CartComplex(a.mul_add(c, -b * d), a.mul_add(d, b * c))
            }
        }

        impl Mul<CartComplex<$t>> for $t {
            type Output = CartComplex<$t>;

            fn mul(self, rhs: CartComplex<$t>) -> Self::Output {
                let CartComplex(a, b) = rhs;
                CartComplex(self * a, self * b)
            }
        }

        impl CartComplex<$t> {
            #[must_use]
            /// Converts a polar form complex number to cartesian form
            pub fn from_polar(polar: PolarComplex<$t>) -> Self {
                let PolarComplex(r, theta) = polar;
                CartComplex(r * <$t>::cos(theta), r * <$t>::sin(theta))
            }

            #[must_use]
            /// Converts a tuple of numbers to a complex number where the first index represents
            /// the real part and the second represents the imaginary.
            pub const fn from(a: $t, b: $t) -> Self {
                CartComplex(a, b)
            }

            #[must_use]
            /// Returns the real part of the complex number
            pub const fn re(&self) -> $t {
                self.0
            }

            #[must_use]
            /// Returns the imaginary part of the complex number
            pub const fn im(&self) -> $t {
                self.1
            }

            #[must_use]
            /// Returns the complex conjugate
            pub fn conj(self) -> Self {
                CartComplex(self.re(), -self.im())
            }

            #[must_use]
            /// Returns the norm (absolute value, hypotenuse, magnitude) of the complex number
            pub fn norm(&self) -> $t {
                self.0.hypot(self.1)
            }
        }
    };
}

/// Implements the logic and functionality for polar complex numbers
macro_rules! impl_polar_complex {
    ($t:ty) => {
        impl Add for PolarComplex<$t> {
            type Output = PolarComplex<$t>;

            fn add(self, rhs: Self) -> Self::Output {
                let lhs_cart = CartComplex::<$t>::from_polar(self);
                let rhs_cart = CartComplex::<$t>::from_polar(rhs);

                let res = lhs_cart + rhs_cart;

                Self::from_cartesian(res)
            }
        }

        impl Mul for PolarComplex<$t> {
            type Output = PolarComplex<$t>;

            fn mul(self, rhs: Self) -> Self::Output {
                let PolarComplex(r1, theta1) = self;
                let PolarComplex(r2, theta2) = rhs;

                PolarComplex(r1 * r2, theta1 + theta2)
            }
        }

        impl Mul<PolarComplex<$t>> for $t {
            type Output = PolarComplex<$t>;

            fn mul(self, rhs: PolarComplex<$t>) -> Self::Output {
                let PolarComplex(r, theta) = rhs;
                PolarComplex(self * r, theta)
            }
        }

        impl PolarComplex<$t> {
            #[must_use]
            /// Converts a cartesian form complex number to polar form
            pub fn from_cartesian(cart: CartComplex<$t>) -> Self {
                let CartComplex(a, b) = cart;
                let theta = (b / a).atan();
                let r = cart.norm();

                PolarComplex(r, theta)
            }

            #[must_use]
            /// Converts a tuple to a polar form complex number. (r, theta) -> r(cos(theta) + i*sin(theta))
            pub const fn from(a: $t, b: $t) -> Self {
                PolarComplex(a, b)
            }

            #[must_use]
            /// Returns the real part of the complex number
            pub fn re(&self) -> $t {
                let PolarComplex(r, theta) = self;
                r * <$t>::cos(*theta)
            }

            #[must_use]
            /// Returns the imaginary part of the complex number
            pub fn im(&self) -> $t {
                let PolarComplex(r, theta) = self;
                r * <$t>::sin(*theta)
            }

            #[must_use]
            /// Returns the complex conjugate
            pub fn conj(self) -> Self {
                let PolarComplex(r, theta) = self;
                PolarComplex(r, -theta)
            }

            #[must_use]
            /// Returns the norm (absolute value, hypotenuse, magnitude) of the complex number
            pub const fn norm(&self) -> $t {
                self.0
            }
        }
    };
}

impl_cart_complex!(f32);
impl_cart_complex!(f64);

impl_polar_complex!(f32);
impl_polar_complex!(f64);

#[cfg(test)]
mod tests {
    use super::{CartComplex, PolarComplex};
    use float_cmp::approx_eq;
    use rand::Rng;

    fn approx_eq_cart(lhs: &CartComplex<f32>, rhs: &CartComplex<f32>) -> bool {
        approx_eq!(f32, lhs.0, rhs.0) && approx_eq!(f32, lhs.1, rhs.1)
    }

    fn approx_eq_polar(lhs: &PolarComplex<f32>, rhs: &PolarComplex<f32>) -> bool {
        approx_eq!(f32, lhs.0, rhs.0) && approx_eq!(f32, lhs.1, rhs.1)
    }

    #[test]
    fn basic() {
        let neg1: CartComplex<f32> = CartComplex(0.0, 1.0);
        let squared: CartComplex<f32> = neg1 * neg1;
        assert!(approx_eq_cart(&squared, &CartComplex(-1.0, 0.0)));
    }

    #[test]
    fn random_identity_verification() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let (a, b, c, d) = (
                rng.gen_range(-1000.0..1000.0),
                rng.gen_range(-1000.0..1000.0),
                rng.gen_range(-1000.0..1000.0),
                rng.gen_range(-1000.0..1000.0),
            );

            assert_identities_cart(a, b, c, d);
        }
    }

    fn assert_identities_cart(a: f32, b: f32, c: f32, d: f32) {
        let z = CartComplex(a, b);
        let w = CartComplex(c, d);

        // z + \bar{z} = 2Re(z);
        let want = CartComplex(2.0 * z.re(), 0.0);
        let got = z + z.conj();
        assert!(approx_eq_cart(&want, &got));

        // z - \bar{z} = 2(Im(z))i;
        let want = CartComplex(0.0, 2.0 * z.im());
        let got = z + (-1.0) * z.conj();

        assert!(approx_eq_cart(&want, &got));

        // z\bar{z} = |z|^2;
        let want = CartComplex(z.norm().powi(2), 0.0);
        let got = z * z.conj();

        assert!(approx_eq_cart(&want, &got));

        // \bar{w + z} = \bar{w} + \bar{z}
        let want = (w + z).conj();
        let got = w.conj() + z.conj();

        assert!(approx_eq_cart(&want, &got));

        // \bar{wz} = \bar{w}\bar{z};
        let want = (w * z).conj();
        let got = w.conj() * z.conj();

        assert!(approx_eq_cart(&want, &got));

        // \bar{\bar{z}} = z;
        let want = z;
        let got = z.conj().conj();

        assert!(approx_eq_cart(&want, &got));

        // |Re(z)| \le |z| and |Im(z)| \le |z|;
        assert!(z.re().abs() <= z.norm());
        assert!(z.im().abs() <= z.norm());

        // |\bar{z}| = |z|
        assert!(approx_eq!(f32, z.conj().norm(), z.norm()));

        // |wz| = |w||z|;
        assert!(approx_eq!(f32, (w * z).norm(), w.norm() * z.norm()));

        // |w + z| \le |w| + |z|;
        assert!((w + z).norm() <= w.norm() + z.norm());
    }

    mod polar {
        use super::*;

        // Define a small epsilon for floating-point comparisons
        const EPSILON: f32 = 1e-6;

        #[test]
        fn test_addition() {
            let z1 = PolarComplex::<f32>::from(1.0_f32, 0.0_f32);
            let z2 = PolarComplex::<f32>::from(1.0, std::f32::consts::FRAC_PI_2); // 90 degrees

            let result = z1 + z2;
            let expected = PolarComplex::<f32>::from((2.0_f32).sqrt(), std::f32::consts::FRAC_PI_4);

            assert!(approx_eq_polar(&result, &expected));
        }

        #[test]
        fn test_multiplication() {
            let z1 = PolarComplex::<f32>::from(2.0, std::f32::consts::FRAC_PI_4); // 45 degrees
            let z2 = PolarComplex::<f32>::from(3.0, std::f32::consts::FRAC_PI_2); // 90 degrees

            let result = z1 * z2;
            let expected = PolarComplex::<f32>::from(6.0, 3.0 * std::f32::consts::FRAC_PI_4); // 135 degrees

            assert!(approx_eq_polar(&result, &expected));
        }

        #[test]
        fn test_scalar_multiplication() {
            let scalar = 2.0;
            let z = PolarComplex::<f32>::from(3.0, std::f32::consts::FRAC_PI_3); // 60 degrees

            let result = scalar * z;
            let expected = PolarComplex::<f32>::from(6.0, std::f32::consts::FRAC_PI_3); // 60 degrees

            assert!(approx_eq_polar(&result, &expected));
        }
        #[test]
        fn test_conjugate() {
            let z = PolarComplex::<f32>::from(4.0, std::f32::consts::FRAC_PI_4); // 45 degrees

            let result = z.conj();
            let expected = PolarComplex::<f32>::from(4.0, -std::f32::consts::FRAC_PI_4); // -45 degrees

            assert!(approx_eq_polar(&result, &expected));
        }
        #[test]
        fn test_from_cartesian() {
            let cartesian = CartComplex::<f32>::from(1.0, 1.0);
            let result = PolarComplex::<f32>::from_cartesian(cartesian);
            let expected = PolarComplex::<f32>::from(2.0_f32.sqrt(), std::f32::consts::FRAC_PI_4); // 45 degrees

            assert!(approx_eq_polar(&result, &expected));
        }
        #[test]
        fn test_norm() {
            let z = PolarComplex::<f32>::from(3.0, std::f32::consts::FRAC_PI_3); // 60 degrees
            let result = z.norm();
            let expected = 3.0;

            assert!(approx_eq!(f32, result, expected, epsilon = EPSILON));
        }
    }
}
