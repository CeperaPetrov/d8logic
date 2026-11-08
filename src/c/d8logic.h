/* d8logic.h -- eight-valued logic on the D8 complex plane.
 * Single header, C99, no dependencies besides <math.h>. Link with -lm.
 *
 * Axes:   first coordinate = logic (Yes / No),   second = emotion (Like / Dislike).
 * Logic is crisp: the first axis gives a definite yes or no. Emotion lives on the
 * second axis. Combinations of logical and emotional decisions may yield
 * irrational decisions.
 * Sectors are half-open [lower, upper), counted counter-clockwise from the +logic axis,
 * so every canonical state is the *lower* ray of its sector:
 *
 *   0 YES          [  0,  45)   +logic axis
 *   1 YES_LIKE     [ 45,  90)   diagonal
 *   2 LIKE         [ 90, 135)   +emotion axis
 *   3 NO_LIKE      [135, 180)   diagonal
 *   4 NO           [180, 225)   -logic axis
 *   5 NO_DISLIKE   [225, 270)   diagonal
 *   6 DISLIKE      [270, 315)   -emotion axis
 *   7 YES_DISLIKE  [315, 360)   diagonal
 *   8 INDIFFERENT  the origin (zero)
 *   9 UNDEFINED    NaN: outside the space (non-finite input)
 *
 * Representation is the D8 one: (x, y) in the wedge 0 <= y <= x plus an 8-bit
 * octant mask (one bit inside an octant, two adjacent bits on a ray).
 * Multiplication is local: no trig, no global Cartesian conversion.
 *
 * The decision is the sector the vector lands in. Magnitude is an optional extra.
 */
#ifndef D8LOGIC_H
#define D8LOGIC_H

#include <math.h>
#include <stdint.h>

typedef enum {
    D8L_YES = 0, D8L_YES_LIKE, D8L_LIKE, D8L_NO_LIKE,
    D8L_NO, D8L_NO_DISLIKE, D8L_DISLIKE, D8L_YES_DISLIKE,
    D8L_INDIFFERENT, D8L_UNDEFINED
} d8l_state;

typedef struct { double x, y; uint8_t mask; } d8l;

static inline uint8_t d8l_rol_(uint8_t m, unsigned n) {
    n &= 7u;
    return (uint8_t)(((unsigned)m << n) | ((unsigned)m >> ((8u - n) & 7u)));
}
static inline uint8_t d8l_ror_(uint8_t m, unsigned n) {
    return d8l_rol_(m, (8u - (n & 7u)) & 7u);
}
static inline int d8l_bit_index_(uint8_t m) {
    int k = 0;
    while ((m >>= 1u) != 0u) ++k;
    return k;
}
static inline int d8l_one_hot_(uint8_t m) { return m != 0 && (m & (uint8_t)(m - 1u)) == 0; }
static inline int d8l_even_anchor_(uint8_t m) {
    if (d8l_one_hot_(m)) return d8l_bit_index_(m);
    for (int k = 0; k < 8; k += 2)
        if (m & (uint8_t)(1u << k)) return k;
    return -1;
}

static inline d8l d8l_zero(void) { d8l r = {0.0, 0.0, 0xFF}; return r; }
static inline d8l d8l_nan(void)  { d8l r = {0.0, 0.0, 0x00}; return r; }

/* Normalized one-bit counter -> final mask (adds 2nd bit when on a ray). */
static inline d8l d8l_reduced_(double x, double y, uint8_t counter) {
    if (x == 0.0 && y == 0.0) return d8l_zero();
    const int k = d8l_bit_index_(counter);
    if (y == 0.0) {
        const int nb = (k & 1) ? ((k + 1) & 7) : ((k + 7) & 7);
        counter = (uint8_t)(counter | (1u << nb));
    } else if (x == y) {
        const int nb = (k & 1) ? ((k + 7) & 7) : ((k + 1) & 7);
        counter = (uint8_t)(counter | (1u << nb));
    }
    d8l r = {x, y, counter};
    return r;
}

/* Cartesian point -> D8. Non-finite input gives the NaN state. */
static inline d8l d8l_from_xy(double logic, double emotion) {
    if (!isfinite(logic) || !isfinite(emotion)) return d8l_nan();
    if (logic == 0.0 && emotion == 0.0) return d8l_zero();
    int q; double u, v;
    if (logic > 0.0 && emotion >= 0.0)      { q = 0; u = logic;  v = emotion;  }
    else if (logic <= 0.0 && emotion > 0.0) { q = 1; u = emotion;  v = -logic; }
    else if (logic < 0.0 && emotion <= 0.0) { q = 2; u = -logic; v = -emotion; }
    else                            { q = 3; u = -emotion; v = logic;  }
    v += 0.0; /* normalise -0.0 */
    if (v < u) return d8l_reduced_(u, v, (uint8_t)(1u << (2 * q)));
    return d8l_reduced_(v, u, (uint8_t)(1u << (2 * q + 1)));
}

/* Canonical point of a state (its lower ray) with |z| == strength. */
static inline d8l d8l_from_state(d8l_state s, double strength) {
    if (s == D8L_UNDEFINED) return d8l_nan();
    if (s == D8L_INDIFFERENT || strength == 0.0) return d8l_zero();
    if ((int)s & 1) {
        const double a = strength * sqrt(0.5);
        return d8l_reduced_(a, a, (uint8_t)(1u << (int)s));
    }
    return d8l_reduced_(strength, 0.0, (uint8_t)(1u << (int)s));
}

static inline int d8l_is_zero(d8l a) { return a.mask == 0xFF; }
static inline int d8l_is_nan(d8l a)  { return a.mask == 0x00; }

/* The decision: which sector the point is in. */
static inline d8l_state d8l_state_of(d8l a) {
    const uint8_t m = a.mask;
    if (m == 0xFF) return D8L_INDIFFERENT;
    if (m == 0x00) return D8L_UNDEFINED;
    const uint8_t t = (uint8_t)(m & d8l_rol_(m, 1)); /* upper bit of a two-bit ray */
    return (d8l_state)d8l_bit_index_(t ? t : m);
}

static inline double d8l_magnitude(d8l a) { return hypot(a.x, a.y); }

/* Back to (logic, emotion); NaNs for the NaN state. */
static inline void d8l_to_xy(d8l a, double *logic, double *emotion) {
    if (a.mask == 0x00) { *logic = *emotion = NAN; return; }
    if (a.mask == 0xFF) { *logic = *emotion = 0.0; return; }
    const int s = (int)d8l_state_of(a);
    const double u = (s & 1) ? a.y : a.x;
    const double v = (s & 1) ? a.x : a.y;
    switch (s >> 1) {
        case 0:  *logic = u;       *emotion = v;       break;
        case 1:  *logic = 0.0 - v; *emotion = u;       break;
        case 2:  *logic = 0.0 - u; *emotion = 0.0 - v; break;
        default: *logic = v;       *emotion = 0.0 - u; break;
    }
}

/* Same direction, x scaled by a power of two into [1, 2) (exact). */
static inline d8l d8l_unit_(d8l z) {
    int e;
    frexp(z.x, &e);
    z.x = ldexp(z.x, 1 - e);
    z.y = ldexp(z.y, 1 - e);
    return z;
}

/* Local product core on finite wedge values: out_x, out_y and a one-bit counter. */
static inline void d8l_product_(d8l a, d8l b, double *ox, double *oy, uint8_t *counter_out) {
    const int ka = d8l_even_anchor_(a.mask), kb = d8l_even_anchor_(b.mask);
    const int qa = ((ka + 1) / 2) & 3, qb = ((kb + 1) / 2) & 3;
    const int pa = ka & 1, pb = kb & 1;

    const double ac = a.x * b.x, bd = a.y * b.y;
    const double ad = a.x * b.y, bc = a.y * b.x;

    uint8_t counter = d8l_rol_(0x01, (unsigned)(2 * ((qa + qb) & 3)));
    if (pa == pb) {
        const int cross = ad + bc + bd > ac;
        if (pa == 0) {
            if (cross) counter = d8l_rol_(counter, 1);
        } else {
            counter = d8l_ror_(counter, 1);
            if (cross) counter = d8l_ror_(counter, 1);
        }
        if (cross) { *ox = ad + bc; *oy = ac - bd; }
        else       { *ox = ac - bd; *oy = ad + bc; }
    } else {
        const int neg_dev = (pa == 0) ? (bc < ad) : (ad < bc);
        if (neg_dev) counter = d8l_ror_(counter, 1);
        *ox = ac + bd;
        *oy = (bc >= ad) ? (bc - ad) : (ad - bc);
    }
    *counter_out = counter;
}

/* Composition: angles add, magnitudes multiply. Zero absorbs, NaN beats zero. */
static inline d8l d8l_mul(d8l a, d8l b) {
    if (a.mask == 0x00 || b.mask == 0x00) return d8l_nan();
    if (a.mask == 0xFF || b.mask == 0xFF) return d8l_zero();
    double ox, oy;
    uint8_t counter;
    /* Extreme scale: keep the sector right. Comment out this block to get plain
     * D8 arithmetic (NaN on overflow), as in d8_complex.hpp. */
    if (!(a.x * b.x > 0x1p-900 && a.x * b.x < 0x1p+900)) {
        a = d8l_unit_(a);
        b = d8l_unit_(b);
    }
    d8l_product_(a, b, &ox, &oy, &counter);
    if (!isfinite(ox) || !isfinite(oy)) return d8l_nan();
    return d8l_reduced_(ox, oy, counter);
}

/* Opposite on both axes (Yes<->No and Like<->Dislike): rotate 180 deg. */
static inline d8l d8l_neg(d8l a) { a.mask = d8l_rol_(a.mask, 4); return a; }

/* Like<->Dislike, logic kept (complex conjugate): mirror the mask. */
static inline d8l d8l_not_emotion(d8l a) {
    uint8_t r = 0;
    for (int k = 0; k < 8; ++k)
        if ((a.mask >> k) & 1u) r = (uint8_t)(r | (1u << (7 - k)));
    a.mask = r;
    return a;
}

/* Yes<->No, emotion kept (mirror across the emotion axis). */
static inline d8l d8l_not_logic(d8l a) { return d8l_neg(d8l_not_emotion(a)); }

/* Superposition of opinions: add in Cartesian form, then re-sector. */
static inline d8l d8l_add(d8l a, d8l b) {
    double ar, ai, br, bi;
    d8l_to_xy(a, &ar, &ai);
    d8l_to_xy(b, &br, &bi);
    double sx = ar + br, sy = ai + bi;
    if ((!isfinite(sx) || !isfinite(sy)) &&
        isfinite(ar) && isfinite(ai) && isfinite(br) && isfinite(bi)) {
        sx = ar * 0.5 + br * 0.5;   /* overflow: halve */
        sy = ai * 0.5 + bi * 0.5;
    }
    return d8l_from_xy(sx, sy);
}

static inline const char *d8l_name(d8l_state s) {
    static const char *const n[] = {
        "YES", "YES_LIKE", "LIKE", "NO_LIKE", "NO",
        "NO_DISLIKE", "DISLIKE", "YES_DISLIKE", "INDIFFERENT", "UNDEFINED"};
    return ((unsigned)s < 10u) ? n[(int)s] : "?";
}

#endif /* D8LOGIC_H */
