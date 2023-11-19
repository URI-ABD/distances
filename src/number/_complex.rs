use float_cmp::approx_eq;
use std::ops::{Add, Mul};

#[derive(Clone, Copy, Debug)]
pub struct CartComplex<T>(T, T);

#[derive(Clone, Copy, Debug)]
pub struct PolarComplex<T>(T, T);

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
                CartComplex(a * c - b * d, a * d + b * c)
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
            pub fn from<G: Into<($t, $t)>>(data: G) -> Self {
                let (a, b) = data.into();
                CartComplex(a, b)
            }

            pub fn re(&self) -> $t {
                self.0
            }

            pub fn im(&self) -> $t {
                self.1
            }

            pub fn conj(self) -> Self {
                CartComplex(self.re(), -self.im())
            }

            pub fn norm(&self) -> $t {
                self.0.hypot(self.1)
            }

            pub fn approx_eq(&self, other: &Self) -> bool {
                approx_eq!($t, self.0, other.0) && approx_eq!($t, self.1, other.1)
            }
        }
    };
}

macro_rules! impl_polar_complex {
    ($t:ty) => {
        impl Add for PolarComplex<$t> {
            type Output = PolarComplex<$t>;

            fn add(self, rhs: Self) -> Self::Output {
                let PolarComplex(r1, theta1) = self;
                let PolarComplex(r2, theta2) = rhs;

                let real_part = r1 * <$t>::cos(theta1) + r2 * <$t>::cos(theta2);
                let imag_part = r1 * <$t>::sin(theta1) + r2 * <$t>::sin(theta2);

                let r_result = real_part.hypot(imag_part);
                let theta_result = imag_part.atan2(real_part);

                PolarComplex(r_result, theta_result)
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
            pub fn from_cartesian(cart: CartComplex<$t>) -> Self {
                let CartComplex(a, b) = cart;
                let theta = (b / a).atan();
                let r = cart.norm();

                PolarComplex(r, theta)
            }
        }

        impl PolarComplex<$t> {
            pub fn from<G: Into<($t, $t)>>(data: G) -> Self {
                let (a, b) = data.into();
                PolarComplex(a, b)
            }

            pub fn re(&self) -> $t {
                let PolarComplex(r, theta) = self;
                r * <$t>::cos(*theta)
            }

            pub fn im(&self) -> $t {
                let PolarComplex(r, theta) = self;
                r * <$t>::sin(*theta)
            }

            // a - bi => (r, theta) where r = |z|, theta = arctan(a/-b) = arctan(-(a/b)) = -arctan(a/b)
            pub fn conj(self) -> Self {
                let PolarComplex(r, theta) = self;
                PolarComplex(r, -theta)
            }

            pub fn norm(&self) -> $t {
                self.0
            }

            pub fn approx_eq(&self, other: &Self) -> bool {
                approx_eq!($t, self.0, other.0) && approx_eq!($t, self.1, other.1)
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
    use crate::CartComplex;
    use float_cmp::approx_eq;
    use rand::Rng;

    #[test]
    fn basic() {
        let neg1 = CartComplex(0.0, 1.0);
        let squared: CartComplex<f64> = neg1 * neg1;
        assert!(squared.approx_eq(&CartComplex(-1.0, 0.0)));
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

        for _ in 0..100 {
            let pi = std::f32::consts::PI;

            let (r1, r2, theta1, theta2) = (
                rng.gen_range(-1000.0..1000.0),
                rng.gen_range(-1000.0..1000.0),
                rng.gen_range(-2.0 * pi..2.0 * pi),
                rng.gen_range(-2.0 * pi..2.0 * pi),
            );

            assert_identities_polar(r1, theta1, r2, theta2);
        }
    }

    fn assert_identities_cart(a: f32, b: f32, c: f32, d: f32) {
        let z = CartComplex(a, b);
        let w = CartComplex(c, d);

        // z + \bar{z} = 2Re(z);
        let want = CartComplex(2.0 * z.re(), 0.0);
        let got = z + z.conj();
        assert!(got.approx_eq(&want));

        // z - \bar{z} = 2(Im(z))i;
        let want = CartComplex(0.0, 2.0 * z.im());
        let got = z + (-1.0) * z.conj();

        assert!(got.approx_eq(&want));

        // z\bar{z} = |z|^2;
        let want = CartComplex(z.norm().powi(2), 0.0);
        let got = z * z.conj();

        assert!(got.approx_eq(&want));

        // \bar{w + z} = \bar{w} + \bar{z}
        let want = (w + z).conj();
        let got = w.conj() + z.conj();

        assert!(got.approx_eq(&want));

        // \bar{wz} = \bar{w}\bar{z};
        let want = (w * z).conj();
        let got = w.conj() * z.conj();

        assert!(got.approx_eq(&want));

        // \bar{\bar{z}} = z;
        let want = z;
        let got = z.conj().conj();

        assert!(got.approx_eq(&want));

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

    fn assert_identities_polar(r1: f32, theta1: f32, r2: f32, theta2: f32) {}
}
