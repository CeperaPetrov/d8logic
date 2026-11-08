//! d8logic.rs -- eight-valued logic on the D8 complex plane. Single file, std only.
//!
//! Axes:   first coordinate = logic (Yes / No),   second = emotion (Like / Dislike).
//! Logic is crisp: the first axis gives a definite yes or no. Emotion lives on the
//! second axis. Combinations of logical and emotional decisions may yield
//! irrational decisions.
//! Sectors are half-open [lower, upper), counted counter-clockwise from the +logic axis,
//! so every canonical state is the *lower* ray of its sector:
//!
//! ```text
//!   0 Yes          [  0,  45)   +logic axis
//!   1 YesLike      [ 45,  90)   diagonal
//!   2 Like         [ 90, 135)   +emotion axis
//!   3 NoLike       [135, 180)   diagonal
//!   4 No           [180, 225)   -logic axis
//!   5 NoDislike    [225, 270)   diagonal
//!   6 Dislike      [270, 315)   -emotion axis
//!   7 YesDislike   [315, 360)   diagonal
//!   8 Indifferent  the origin (zero)
//!   9 Undefined    NaN: outside the space (non-finite input)
//! ```
//!
//! Representation is the D8 one: (x, y) in the wedge 0 <= y <= x plus an 8-bit
//! octant mask (one bit inside an octant, two adjacent bits on a ray).
//! Multiplication is local: no trig, no global Cartesian conversion.
//!
//! The decision is the sector the vector lands in. Magnitude is an optional extra.

use std::ops::{Add, Mul, Neg};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum State {
    Yes = 0,
    YesLike = 1,
    Like = 2,
    NoLike = 3,
    No = 4,
    NoDislike = 5,
    Dislike = 6,
    YesDislike = 7,
    Indifferent = 8,
    Undefined = 9,
}

impl State {
    const ALL: [State; 8] = [
        State::Yes, State::YesLike, State::Like, State::NoLike,
        State::No, State::NoDislike, State::Dislike, State::YesDislike,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct D8 {
    pub x: f64,
    pub y: f64,
    pub mask: u8,
}

#[inline]
fn rol(m: u8, n: u32) -> u8 { m.rotate_left(n & 7) }
#[inline]
fn ror(m: u8, n: u32) -> u8 { m.rotate_right(n & 7) }
#[inline]
fn bit_index(m: u8) -> i32 { 7 - m.leading_zeros() as i32 } // m != 0
#[inline]
fn one_hot(m: u8) -> bool { m != 0 && m & (m - 1) == 0 }

fn even_anchor(m: u8) -> i32 {
    if one_hot(m) {
        return bit_index(m);
    }
    for k in (0..8).step_by(2) {
        if m & (1u8 << k) != 0 {
            return k;
        }
    }
    -1
}

/// 2^n exactly, for n in -1022..=1023.
#[inline]
fn pow2(n: i32) -> f64 { f64::from_bits(((1023 + n) as u64) << 52) }

/// e such that x * 2^-e lies in [1, 2); x finite and > 0.
fn exp_of(x: f64) -> i32 {
    let (x, adj) = if x < f64::MIN_POSITIVE { (x * pow2(64), -64) } else { (x, 0) };
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023 + adj
}

/// x * 2^n, done in two exact halves so the scale itself never overflows.
fn scale2(x: f64, n: i32) -> f64 {
    let h = n / 2;
    x * pow2(h) * pow2(n - h)
}

/// Same direction, x scaled by a power of two into [1, 2) (exact).
fn unit(z: D8) -> D8 {
    let n = -exp_of(z.x);
    D8 { x: scale2(z.x, n), y: scale2(z.y, n), mask: z.mask }
}

/// Local product core on finite wedge values: (out_x, out_y, one-bit counter).
fn product(a: D8, b: D8) -> (f64, f64, u8) {
    let ka = even_anchor(a.mask);
    let kb = even_anchor(b.mask);
    let qa = ((ka + 1) / 2) & 3;
    let qb = ((kb + 1) / 2) & 3;
    let pa = ka & 1;
    let pb = kb & 1;

    let ac = a.x * b.x;
    let bd = a.y * b.y;
    let ad = a.x * b.y;
    let bc = a.y * b.x;

    let mut counter = rol(0x01, (2 * ((qa + qb) & 3)) as u32);
    if pa == pb {
        let cross = ad + bc + bd > ac;
        if pa == 0 {
            if cross {
                counter = rol(counter, 1);
            }
        } else {
            counter = ror(counter, 1);
            if cross {
                counter = ror(counter, 1);
            }
        }
        if cross {
            (ad + bc, ac - bd, counter)
        } else {
            (ac - bd, ad + bc, counter)
        }
    } else {
        let neg_dev = if pa == 0 { bc < ad } else { ad < bc };
        if neg_dev {
            counter = ror(counter, 1);
        }
        (ac + bd, if bc >= ad { bc - ad } else { ad - bc }, counter)
    }
}

/// Normalized one-bit counter -> final mask (adds 2nd bit when on a ray).
fn reduced(x: f64, y: f64, mut counter: u8) -> D8 {
    if x == 0.0 && y == 0.0 {
        return D8::ZERO;
    }
    let k = bit_index(counter);
    if y == 0.0 {
        let nb = if k & 1 != 0 { (k + 1) & 7 } else { (k + 7) & 7 };
        counter |= 1u8 << nb;
    } else if x == y {
        let nb = if k & 1 != 0 { (k + 7) & 7 } else { (k + 1) & 7 };
        counter |= 1u8 << nb;
    }
    D8 { x, y, mask: counter }
}

impl D8 {
    pub const ZERO: D8 = D8 { x: 0.0, y: 0.0, mask: 0xFF };
    pub const NAN: D8 = D8 { x: 0.0, y: 0.0, mask: 0x00 };

    /// Cartesian point -> D8. Non-finite input gives the NaN state.
    pub fn from_xy(logic: f64, emotion: f64) -> D8 {
        if !logic.is_finite() || !emotion.is_finite() {
            return D8::NAN;
        }
        if logic == 0.0 && emotion == 0.0 {
            return D8::ZERO;
        }
        let (q, u, mut v) = if logic > 0.0 && emotion >= 0.0 {
            (0u32, logic, emotion)
        } else if logic <= 0.0 && emotion > 0.0 {
            (1, emotion, -logic)
        } else if logic < 0.0 && emotion <= 0.0 {
            (2, -logic, -emotion)
        } else {
            (3, -emotion, logic)
        };
        v += 0.0; // normalise -0.0
        if v < u {
            reduced(u, v, 1u8 << (2 * q))
        } else {
            reduced(v, u, 1u8 << (2 * q + 1))
        }
    }

    /// Canonical point of a state (its lower ray) with |z| == strength.
    pub fn from_state(s: State, strength: f64) -> D8 {
        match s {
            State::Undefined => D8::NAN,
            State::Indifferent => D8::ZERO,
            _ if strength == 0.0 => D8::ZERO,
            _ => {
                let k = s as u8;
                if k & 1 != 0 {
                    let a = strength * 0.5f64.sqrt();
                    reduced(a, a, 1u8 << k)
                } else {
                    reduced(strength, 0.0, 1u8 << k)
                }
            }
        }
    }

    pub fn is_zero(&self) -> bool { self.mask == 0xFF }
    pub fn is_nan(&self) -> bool { self.mask == 0x00 }

    /// The decision: which sector the point is in.
    pub fn state(&self) -> State {
        let m = self.mask;
        match m {
            0xFF => State::Indifferent,
            0x00 => State::Undefined,
            _ => {
                let t = m & rol(m, 1); // upper bit of a two-bit ray
                State::ALL[bit_index(if t != 0 { t } else { m }) as usize]
            }
        }
    }

    pub fn magnitude(&self) -> f64 { self.x.hypot(self.y) }

    /// Back to (logic, emotion); NaNs for the NaN state.
    pub fn to_xy(&self) -> (f64, f64) {
        if self.mask == 0x00 {
            return (f64::NAN, f64::NAN);
        }
        if self.mask == 0xFF {
            return (0.0, 0.0);
        }
        let s = self.state() as u8;
        let (u, v) = if s & 1 != 0 { (self.y, self.x) } else { (self.x, self.y) };
        match s >> 1 {
            0 => (u, v),
            1 => (0.0 - v, u),
            2 => (0.0 - u, 0.0 - v),
            _ => (v, 0.0 - u),
        }
    }

    /// Like<->Dislike, logic kept (complex conjugate): mirror the mask.
    pub fn not_emotion(self) -> D8 {
        D8 { mask: self.mask.reverse_bits(), ..self }
    }

    /// Yes<->No, emotion kept (mirror across the emotion axis).
    pub fn not_logic(self) -> D8 { -self.not_emotion() }
}

/// Composition: angles add, magnitudes multiply. Zero absorbs, NaN beats zero.
impl Mul for D8 {
    type Output = D8;
    fn mul(self, o: D8) -> D8 {
        if self.is_nan() || o.is_nan() {
            return D8::NAN;
        }
        if self.is_zero() || o.is_zero() {
            return D8::ZERO;
        }
        let (mut a, mut b) = (self, o);
        // Extreme scale: keep the sector right. Comment out this block to get plain
        // D8 arithmetic (NaN on overflow), as in d8_complex.hpp.
        if !(a.x * b.x > pow2(-900) && a.x * b.x < pow2(900)) {
            a = unit(a);
            b = unit(b);
        }
        let (ox, oy, counter) = product(a, b);
        if !ox.is_finite() || !oy.is_finite() {
            return D8::NAN;
        }
        reduced(ox, oy, counter)
    }
}

/// Opposite on both axes (Yes<->No and Like<->Dislike): rotate 180 deg.
impl Neg for D8 {
    type Output = D8;
    fn neg(self) -> D8 { D8 { mask: rol(self.mask, 4), ..self } }
}

/// Superposition of opinions: add in Cartesian form, then re-sector.
impl Add for D8 {
    type Output = D8;
    fn add(self, o: D8) -> D8 {
        let (ar, ai) = self.to_xy();
        let (br, bi) = o.to_xy();
        let (mut sx, mut sy) = (ar + br, ai + bi);
        if (!sx.is_finite() || !sy.is_finite())
            && ar.is_finite() && ai.is_finite() && br.is_finite() && bi.is_finite()
        {
            sx = ar * 0.5 + br * 0.5; // overflow: halve
            sy = ai * 0.5 + bi * 0.5;
        }
        D8::from_xy(sx, sy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_sector(logic: f64, emotion: f64) -> u8 {
        // independent reference: signs and one comparison, nothing else
        if logic == 0.0 && emotion == 0.0 {
            return 8;
        }
        let (q, u, v): (u8, f64, f64) = if logic > 0.0 && emotion >= 0.0 {
            (0, logic, emotion)
        } else if logic <= 0.0 && emotion > 0.0 {
            (1, emotion, -logic)
        } else if logic < 0.0 && emotion <= 0.0 {
            (2, -logic, -emotion)
        } else {
            (3, -emotion, logic)
        };
        2 * q + if v < u { 0 } else { 1 }
    }

    #[test]
    fn grid() {
        for a in -6..=6 {
            for b in -6..=6 {
                let (fa, fb) = (a as f64, b as f64);
                let za = D8::from_xy(fa, fb);
                assert_eq!(za.state() as u8, ref_sector(fa, fb));
                assert_eq!(za.not_emotion().state() as u8, ref_sector(fa, -fb));
                assert_eq!((-za).state() as u8, ref_sector(-fa, -fb));
                assert_eq!(za.not_logic().state() as u8, ref_sector(-fa, fb));
                for c in -6..=6 {
                    for d in -6..=6 {
                        let (fc, fd) = (c as f64, d as f64);
                        let zb = D8::from_xy(fc, fd);
                        assert_eq!((za * zb).state() as u8, ref_sector(fa * fc - fb * fd, fa * fd + fb * fc));
                        assert_eq!((za + zb).state() as u8, ref_sector(fa + fc, fb + fd));
                    }
                }
            }
        }
    }

    #[test]
    fn basics() {
        let like = D8::from_state(State::Like, 1.0);
        assert_eq!((like * like).state(), State::No);
        assert_eq!(D8::from_xy(f64::NAN, 1.0).state(), State::Undefined);
    }
}
