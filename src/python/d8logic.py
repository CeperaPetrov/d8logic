"""d8logic.py -- eight-valued logic on the D8 complex plane. Single file, no dependencies.

Axes:   first coordinate = logic (Yes / No),   second = emotion (Like / Dislike).
Logic is crisp: the first axis gives a definite yes or no. Emotion lives on the
second axis. Combinations of logical and emotional decisions may yield
irrational decisions.
Sectors are half-open [lower, upper), counted counter-clockwise from the +logic axis,
so every canonical state is the *lower* ray of its sector:

    0 YES          [  0,  45)   +logic axis
    1 YES_LIKE     [ 45,  90)   diagonal
    2 LIKE         [ 90, 135)   +emotion axis
    3 NO_LIKE      [135, 180)   diagonal
    4 NO           [180, 225)   -logic axis
    5 NO_DISLIKE   [225, 270)   diagonal
    6 DISLIKE      [270, 315)   -emotion axis
    7 YES_DISLIKE  [315, 360)   diagonal
    8 INDIFFERENT  the origin (zero)
    9 UNDEFINED    NaN: outside the space (non-finite input)

Representation is the D8 one: (x, y) in the wedge 0 <= y <= x plus an 8-bit
octant mask (one bit inside an octant, two adjacent bits on a ray).
Multiplication is local: no trig, no global Cartesian conversion.

The decision is the sector the vector lands in. Magnitude is an optional extra.
"""
import math
from enum import IntEnum


class State(IntEnum):
    YES = 0
    YES_LIKE = 1
    LIKE = 2
    NO_LIKE = 3
    NO = 4
    NO_DISLIKE = 5
    DISLIKE = 6
    YES_DISLIKE = 7
    INDIFFERENT = 8
    UNDEFINED = 9


def _rol(m, n=1):
    n &= 7
    return ((m << n) | (m >> ((8 - n) & 7))) & 0xFF


def _ror(m, n=1):
    return _rol(m, (8 - (n & 7)) & 7)


def _bit_index(m):
    return m.bit_length() - 1


def _is_one_hot(m):
    return m != 0 and (m & (m - 1)) == 0


def _even_anchor(m):
    if _is_one_hot(m):
        return _bit_index(m)
    for k in range(0, 8, 2):
        if m & (1 << k):
            return k
    return -1


def _reduced(x, y, counter):
    """Normalized one-bit counter -> final mask (adds 2nd bit on a ray)."""
    if x == 0.0 and y == 0.0:
        return D8.zero()
    k = _bit_index(counter)
    if y == 0.0:
        nb = (k + 1) & 7 if k & 1 else (k + 7) & 7
        counter |= 1 << nb
    elif x == y:
        nb = (k + 7) & 7 if k & 1 else (k + 1) & 7
        counter |= 1 << nb
    return D8(x, y, counter)


_LO, _HI = 2.0 ** -900, 2.0 ** 900


def _unit(z):
    """Same direction, x scaled by a power of two into [1, 2) (exact)."""
    e = math.frexp(z.x)[1]
    return D8(math.ldexp(z.x, 1 - e), math.ldexp(z.y, 1 - e), z.mask)


def _product(a, b):
    """Local product core on finite wedge values: (out_x, out_y, one-bit counter)."""
    ka, kb = _even_anchor(a.mask), _even_anchor(b.mask)
    qa, qb = ((ka + 1) // 2) & 3, ((kb + 1) // 2) & 3
    pa, pb = ka & 1, kb & 1

    ac, bd = a.x * b.x, a.y * b.y
    ad, bc = a.x * b.y, a.y * b.x

    counter = _rol(0x01, 2 * ((qa + qb) & 3))
    if pa == pb:
        cross = ad + bc + bd > ac
        if pa == 0:
            if cross:
                counter = _rol(counter)
        else:
            counter = _ror(counter)
            if cross:
                counter = _ror(counter)
        if cross:
            return ad + bc, ac - bd, counter
        return ac - bd, ad + bc, counter
    neg_dev = (bc < ad) if pa == 0 else (ad < bc)
    if neg_dev:
        counter = _ror(counter)
    return ac + bd, (bc - ad) if bc >= ad else (ad - bc), counter


class D8:
    __slots__ = ("x", "y", "mask")

    def __init__(self, x=0.0, y=0.0, mask=0xFF):
        self.x, self.y, self.mask = x, y, mask

    # ---- constructors -------------------------------------------------
    @staticmethod
    def zero():
        return D8(0.0, 0.0, 0xFF)

    @staticmethod
    def nan():
        return D8(0.0, 0.0, 0x00)

    @staticmethod
    def from_xy(logic, emotion):
        """Cartesian point -> D8. Non-finite input gives the NaN state."""
        if not (math.isfinite(logic) and math.isfinite(emotion)):
            return D8.nan()
        if logic == 0.0 and emotion == 0.0:
            return D8.zero()
        # quadrant by half-open rule: [0,90) [90,180) [180,270) [270,360)
        if logic > 0.0 and emotion >= 0.0:
            q, u, v = 0, logic, emotion
        elif logic <= 0.0 and emotion > 0.0:
            q, u, v = 1, emotion, -logic
        elif logic < 0.0 and emotion <= 0.0:
            q, u, v = 2, -logic, -emotion
        else:
            q, u, v = 3, -emotion, logic
        v += 0.0  # normalise -0.0
        if v < u:
            return _reduced(u, v, 1 << (2 * q))
        return _reduced(v, u, 1 << (2 * q + 1))

    @staticmethod
    def from_state(state, strength=1.0):
        """Canonical point of a state (its lower ray) with |z| == strength."""
        s = int(state)
        if s == State.UNDEFINED:
            return D8.nan()
        if s == State.INDIFFERENT or strength == 0.0:
            return D8.zero()
        if s & 1:
            a = strength * math.sqrt(0.5)
            return _reduced(a, a, 1 << s)
        return _reduced(strength, 0.0, 1 << s)

    # ---- inspection ---------------------------------------------------
    @property
    def state(self):
        m = self.mask
        if m == 0xFF:
            return State.INDIFFERENT
        if m == 0x00:
            return State.UNDEFINED
        t = m & _rol(m)          # two-bit ray: the upper bit has its neighbour below
        return State(_bit_index(t if t else m))

    @property
    def magnitude(self):
        return math.hypot(self.x, self.y)

    def to_xy(self):
        """Back to (logic, emotion); (nan, nan) for the NaN state."""
        if self.mask == 0x00:
            return (math.nan, math.nan)
        if self.mask == 0xFF:
            return (0.0, 0.0)
        s = int(self.state)
        q = s >> 1
        u, v = (self.x, self.y) if s % 2 == 0 else (self.y, self.x)
        return ((u, v), (0.0 - v, u), (0.0 - u, 0.0 - v), (v, 0.0 - u))[q]

    # ---- operations ---------------------------------------------------
    def __mul__(self, o):
        """Composition: angles add, magnitudes multiply. Zero absorbs, NaN beats zero."""
        if self.mask == 0x00 or o.mask == 0x00:
            return D8.nan()
        if self.mask == 0xFF or o.mask == 0xFF:
            return D8.zero()
        a, b = self, o
        # Extreme scale: keep the sector right. Comment out these two lines to get
        # plain D8 arithmetic (NaN on overflow), as in d8_complex.hpp.
        if not (_LO < self.x * o.x < _HI):
            a, b = _unit(self), _unit(o)
        ox, oy, counter = _product(a, b)
        if not (math.isfinite(ox) and math.isfinite(oy)):
            return D8.nan()
        return _reduced(ox, oy, counter)

    def __neg__(self):
        """Opposite on both axes (Yes<->No and Like<->Dislike): rotate 180 deg."""
        return D8(self.x, self.y, _rol(self.mask, 4))

    def not_emotion(self):
        """Like<->Dislike, logic kept (complex conjugate): mirror the mask."""
        m = self.mask
        r = 0
        for k in range(8):
            if m >> k & 1:
                r |= 1 << (7 - k)
        return D8(self.x, self.y, r)

    def not_logic(self):
        """Yes<->No, emotion kept (mirror across the emotion axis)."""
        return -self.not_emotion()

    def __add__(self, o):
        """Superposition of opinions (adds in Cartesian form, then re-sectors)."""
        a, b = self.to_xy(), o.to_xy()
        sx, sy = a[0] + b[0], a[1] + b[1]
        if not (math.isfinite(sx) and math.isfinite(sy)) and all(map(math.isfinite, a + b)):
            sx, sy = a[0] * 0.5 + b[0] * 0.5, a[1] * 0.5 + b[1] * 0.5   # overflow: halve
        return D8.from_xy(sx, sy)

    def __repr__(self):
        return f"D8({self.state.name}, |z|={self.magnitude:.6g})"


if __name__ == "__main__":
    yes, like = D8.from_state(State.YES), D8.from_state(State.LIKE)
    print(yes * like, like * like, -like, (yes + like), D8.from_xy(-3, -1))
