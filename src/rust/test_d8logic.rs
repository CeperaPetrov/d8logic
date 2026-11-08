//! Usage examples and tests for d8logic.rs (the file must sit next to this one).
//!
//! Build and run:
//!   rustc --edition 2021 --test test_d8logic.rs -o test_d8logic && ./test_d8logic
//! In a cargo project: put d8logic.rs into src/ and this file into tests/ with
//! `#[path = "../src/d8logic.rs"] mod d8logic;`.

#[allow(dead_code)]
mod d8logic;

use d8logic::{State, D8};

fn state_of(logic: f64, emotion: f64) -> State {
    D8::from_xy(logic, emotion).state()
}

fn canon(s: State) -> D8 {
    D8::from_state(s, 1.0)
}

fn sign_state(v: f64) -> State {
    if v > 0.0 {
        State::Yes
    } else if v < 0.0 {
        State::No
    } else {
        State::Indifferent
    }
}

const RING: [State; 8] = [
    State::Yes, State::YesLike, State::Like, State::NoLike,
    State::No, State::NoDislike, State::Dislike, State::YesDislike,
];

#[test]
fn canonical_states_round_trip() {
    for s in RING {
        assert_eq!(D8::from_state(s, 1.0).state(), s);
        assert_eq!(D8::from_state(s, 0.25).state(), s);
    }
}

#[test]
fn half_open_sectors() {
    assert_eq!(state_of(1.0, 0.0), State::Yes); // lower border is included
    assert_eq!(state_of(1.0, 1.0), State::YesLike);
    assert_eq!(state_of(0.0, 1.0), State::Like);
    assert_eq!(state_of(-1.0, 1.0), State::NoLike);
    assert_eq!(state_of(-1.0, 0.0), State::No);
    assert_eq!(state_of(-1.0, -1.0), State::NoDislike);
    assert_eq!(state_of(0.0, -1.0), State::Dislike);
    assert_eq!(state_of(1.0, -1.0), State::YesDislike);
    // the asymmetry is by design
    assert_eq!(state_of(1.0, 1e-300), State::Yes);
    assert_eq!(state_of(1.0, -1e-300), State::YesDislike);
}

#[test]
fn zero_and_outside_the_space() {
    assert_eq!(state_of(0.0, 0.0), State::Indifferent); // indifference is exactly zero
    assert_eq!(state_of(-0.0, 0.0), State::Indifferent);
    assert_eq!(state_of(f64::NAN, 1.0), State::Undefined);
    assert_eq!(state_of(f64::INFINITY, 1.0), State::Undefined); // non-finite input
    assert_eq!((D8::NAN * canon(State::Yes)).state(), State::Undefined);
}

// The decision is the sector: scale must never destroy it.

#[test]
fn overflow_keeps_the_decision() {
    let big = D8::from_xy(1e200, 1e200); // 45 degrees
    assert_eq!((big * big).state(), State::Like); // 90 degrees, strength out of range
    let huge = D8::from_xy(1.7e308, 0.0);
    assert_eq!((huge + huge).state(), State::Yes);
}

#[test]
fn underflow_is_not_indifference() {
    let tiny = D8::from_xy(1e-200, 1e-200);
    assert_eq!((tiny * D8::from_xy(1e-200, 0.0)).state(), State::YesLike);
    assert_eq!((tiny * tiny).state(), State::Like);
}

fn exact_sector(re: i32, im: i32) -> usize {
    // integer arithmetic, no floats
    let (q, u, v) = if re > 0 && im >= 0 {
        (0, re, im)
    } else if re <= 0 && im > 0 {
        (1, im, -re)
    } else if re < 0 && im <= 0 {
        (2, -re, -im)
    } else {
        (3, -im, re)
    };
    2 * q + if v < u { 0 } else { 1 }
}

fn ldexp(x: f64, n: i32) -> f64 {
    x * 2f64.powi(n / 2) * 2f64.powi(n - n / 2)
}

#[test]
fn scaled_products_match_exact_arithmetic() {
    let scales = [-1070, -600, 0, 600, 1020]; // 2^-1070 .. 2^1020
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                for d in -2..=2 {
                    if (a, b) == (0, 0) || (c, d) == (0, 0) {
                        continue;
                    }
                    let expected = RING[exact_sector(a * c - b * d, a * d + b * c)];
                    for s1 in scales {
                        for s2 in scales {
                            let za = D8::from_xy(ldexp(a as f64, s1), ldexp(b as f64, s1));
                            let zb = D8::from_xy(ldexp(c as f64, s2), ldexp(d as f64, s2));
                            assert_eq!((za * zb).state(), expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn logic_axis_is_ternary() {
    let vals = [-2.0, 0.0, 3.0];
    for a in vals {
        for b in vals {
            let (za, zb) = (D8::from_xy(a, 0.0), D8::from_xy(b, 0.0));
            assert_eq!((za * zb).state(), sign_state(a * b)); // sign product, 0 absorbs
            assert_eq!((za + zb).state(), sign_state(a + b)); // weighted vote
        }
    }
}

#[test]
fn composition_of_canonical_states() {
    let (yes, like) = (canon(State::Yes), canon(State::Like));
    let (dislike, no) = (canon(State::Dislike), canon(State::No));
    let yl = canon(State::YesLike);
    assert_eq!((yes * like).state(), State::Like); // YES is the identity
    assert_eq!((like * like).state(), State::No);
    assert_eq!((like * dislike).state(), State::Yes);
    assert_eq!((no * no).state(), State::Yes);
    assert_eq!((yl * yl).state(), State::Like); // 45 + 45 = 90
}

#[test]
fn absorbing_elements() {
    let x = canon(State::Like);
    assert_eq!((D8::ZERO * x).state(), State::Indifferent);
    assert_eq!((D8::NAN * x).state(), State::Undefined);
    assert_eq!((D8::NAN * D8::ZERO).state(), State::Undefined); // NaN beats zero
}

#[test]
fn hidden_phase_inside_a_sector() {
    // (3, 2) is in sector YES; composed with itself it is (5, 12): sector 1
    let a = D8::from_xy(3.0, 2.0);
    assert_eq!(a.state(), State::Yes);
    assert_eq!((a * a).state(), State::YesLike);
}

#[test]
fn magnitude_is_strength() {
    let (a, b) = (D8::from_state(State::No, 0.5), D8::from_state(State::Like, 4.0));
    assert!((a.magnitude() - 0.5).abs() < 1e-12);
    assert!(((a * b).magnitude() - 2.0).abs() < 1e-12);
}

#[test]
fn negations_on_canonical_states() {
    for j in 0..8 {
        let z = canon(RING[j]);
        assert_eq!((-z).state(), RING[(j + 4) % 8]); // both axes flip
        assert_eq!(z.not_emotion().state(), RING[(8 - j) % 8]); // keep logic
        assert_eq!(z.not_logic().state(), RING[(4 + 8 - j) % 8]); // keep emotion
    }
}

#[test]
fn superposition_and_to_xy() {
    assert_eq!((canon(State::Yes) + canon(State::Like)).state(), State::YesLike);
    assert_eq!((canon(State::Yes) + canon(State::No)).state(), State::Indifferent);
    assert_eq!(D8::from_state(State::No, 2.0).to_xy(), (-2.0, 0.0));
    assert_eq!(D8::from_state(State::Dislike, 2.0).to_xy(), (0.0, -2.0));
}
