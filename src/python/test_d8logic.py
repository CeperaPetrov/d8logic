"""Usage examples and tests for d8logic.py (the file must sit next to this one).

Run:  python3 test_d8logic.py        (plain asserts, no dependencies)
  or: pytest test_d8logic.py
"""
import math

from d8logic import D8, State


def state_of(logic, emotion):
    return D8.from_xy(logic, emotion).state


# --- reading a decision -------------------------------------------------

def test_canonical_states_round_trip():
    for s in range(8):
        assert D8.from_state(State(s)).state == State(s)
        assert D8.from_state(State(s), 0.25).state == State(s)


def test_half_open_sectors():
    # the lower border belongs to the sector: +logic axis is YES
    assert state_of(1, 0) == State.YES
    assert state_of(1, 1) == State.YES_LIKE          # diagonal -> sector 1
    assert state_of(0, 1) == State.LIKE              # +emotion axis -> sector 2
    assert state_of(-1, 1) == State.NO_LIKE
    assert state_of(-1, 0) == State.NO
    assert state_of(-1, -1) == State.NO_DISLIKE
    assert state_of(0, -1) == State.DISLIKE
    assert state_of(1, -1) == State.YES_DISLIKE


def test_asymmetry_is_by_design():
    # (1, +eps) stays YES, (1, -eps) is already YES and dislike
    assert state_of(1, 1e-300) == State.YES
    assert state_of(1, -1e-300) == State.YES_DISLIKE


def test_zero_and_outside_the_space():
    assert state_of(0, 0) == State.INDIFFERENT          # indifference is exactly zero
    assert state_of(-0.0, 0.0) == State.INDIFFERENT
    assert state_of(math.nan, 1) == State.UNDEFINED     # NaN: no direction
    assert state_of(math.inf, 1) == State.UNDEFINED     # non-finite input
    assert (D8.nan() * D8.from_state(State.YES)).state == State.UNDEFINED


# --- the decision is the sector: scale must never destroy it -----------------

def test_overflow_keeps_the_decision():
    big = D8.from_xy(1e200, 1e200)                       # 45 degrees
    assert (big * big).state == State.LIKE               # 90 degrees, strength out of range
    huge = D8.from_xy(1.7e308, 0)
    assert (huge + huge).state == State.YES              # the sum overflows, the answer does not


def test_underflow_is_not_indifference():
    tiny = D8.from_xy(1e-200, 1e-200)
    assert (tiny * D8.from_xy(1e-200, 0)).state == State.YES_LIKE
    assert (tiny * tiny).state == State.LIKE


def _exact_sector(re, im):                               # integer arithmetic, no floats
    if re > 0 and im >= 0: q, u, v = 0, re, im
    elif re <= 0 and im > 0: q, u, v = 1, im, -re
    elif re < 0 and im <= 0: q, u, v = 2, -re, -im
    else: q, u, v = 3, -im, re
    return 2 * q + (0 if v < u else 1)


def test_scaled_products_match_exact_arithmetic():
    pts = [(a, b) for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0)]
    scales = (-1070, -600, 0, 600, 1020)                 # 2^-1070 .. 2^1020
    for (a, b) in pts:
        for (c, d) in pts:
            expected = _exact_sector(a * c - b * d, a * d + b * c)
            for s1 in scales:
                for s2 in scales:
                    za = D8.from_xy(math.ldexp(a, s1), math.ldexp(b, s1))
                    zb = D8.from_xy(math.ldexp(c, s2), math.ldexp(d, s2))
                    assert int((za * zb).state) == expected


# --- logic axis alone: ternary logic ------------------------------------

def sign_state(v):
    return State.YES if v > 0 else State.NO if v < 0 else State.INDIFFERENT


def test_logic_axis_is_ternary():
    vals = (-2.0, 0.0, 3.0)
    for a in vals:
        for b in vals:
            za, zb = D8.from_xy(a, 0), D8.from_xy(b, 0)
            assert (za * zb).state == sign_state(a * b)   # sign product, 0 absorbs
            assert (za + zb).state == sign_state(a + b)   # weighted vote


# --- composition (multiplication) ---------------------------------------

def test_composition_of_canonical_states():
    yes, like = D8.from_state(State.YES), D8.from_state(State.LIKE)
    dislike, no = D8.from_state(State.DISLIKE), D8.from_state(State.NO)
    assert (yes * like).state == State.LIKE          # YES is the identity
    assert (like * like).state == State.NO
    assert (like * dislike).state == State.YES
    assert (no * no).state == State.YES
    yl = D8.from_state(State.YES_LIKE)
    assert (yl * yl).state == State.LIKE             # two 45-degree steps = 90


def test_absorbing_elements():
    x = D8.from_state(State.LIKE)
    assert (D8.zero() * x).state == State.INDIFFERENT
    assert (D8.nan() * x).state == State.UNDEFINED
    assert (D8.nan() * D8.zero()).state == State.UNDEFINED   # NaN beats zero


def test_hidden_phase_inside_a_sector():
    # (3, 2) is in sector YES; composed with itself it is (5, 12): sector 1
    a = D8.from_xy(3, 2)
    assert a.state == State.YES
    assert (a * a).state == State.YES_LIKE


def test_magnitude_is_strength():
    a, b = D8.from_state(State.NO, 0.5), D8.from_state(State.LIKE, 4.0)
    assert math.isclose(a.magnitude, 0.5)
    assert math.isclose((a * b).magnitude, 2.0)


# --- negations -----------------------------------------------------------

def test_negations_on_canonical_states():
    for j in range(8):
        z = D8.from_state(State(j))
        assert (-z).state == State((j + 4) % 8)              # both axes flip
        assert z.not_emotion().state == State((8 - j) % 8)   # keep logic
        assert z.not_logic().state == State((4 - j) % 8)     # keep emotion


# --- superposition and conversion ---------------------------------------

def test_superposition():
    yes, like = D8.from_state(State.YES), D8.from_state(State.LIKE)
    assert (yes + like).state == State.YES_LIKE
    assert (yes + D8.from_state(State.NO)).state == State.INDIFFERENT


def test_to_xy_on_axes_is_exact():
    assert D8.from_state(State.YES, 2).to_xy() == (2.0, 0.0)
    assert D8.from_state(State.NO, 2).to_xy() == (-2.0, 0.0)
    assert D8.from_state(State.LIKE, 2).to_xy() == (0.0, 2.0)
    assert D8.from_state(State.DISLIKE, 2).to_xy() == (0.0, -2.0)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} tests passed")
