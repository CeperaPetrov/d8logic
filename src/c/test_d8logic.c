/* Usage examples and tests for d8logic.h (the header must sit next to this file).
 *
 * Build and run:
 *   cc -std=c99 -Wall -Wextra -O2 test_d8logic.c -o test_d8logic -lm && ./test_d8logic
 */
#include <stdio.h>
#include <math.h>
#include "d8logic.h"

static int failures = 0, checks = 0;
#define CHECK(cond) do { ++checks; if (!(cond)) { ++failures; \
    printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)

static d8l_state state_of_xy(double logic, double emotion) {
    return d8l_state_of(d8l_from_xy(logic, emotion));
}
static d8l canon(d8l_state s) { return d8l_from_state(s, 1.0); }
static d8l_state sign_state(double v) {
    return v > 0 ? D8L_YES : v < 0 ? D8L_NO : D8L_INDIFFERENT;
}

static void test_canonical_states_round_trip(void) {
    for (int s = 0; s < 8; ++s) {
        CHECK(d8l_state_of(d8l_from_state((d8l_state)s, 1.0)) == (d8l_state)s);
        CHECK(d8l_state_of(d8l_from_state((d8l_state)s, 0.25)) == (d8l_state)s);
    }
}

static void test_half_open_sectors(void) {
    CHECK(state_of_xy( 1,  0) == D8L_YES);          /* lower border is included */
    CHECK(state_of_xy( 1,  1) == D8L_YES_LIKE);
    CHECK(state_of_xy( 0,  1) == D8L_LIKE);
    CHECK(state_of_xy(-1,  1) == D8L_NO_LIKE);
    CHECK(state_of_xy(-1,  0) == D8L_NO);
    CHECK(state_of_xy(-1, -1) == D8L_NO_DISLIKE);
    CHECK(state_of_xy( 0, -1) == D8L_DISLIKE);
    CHECK(state_of_xy( 1, -1) == D8L_YES_DISLIKE);
    /* the asymmetry is by design */
    CHECK(state_of_xy(1,  1e-300) == D8L_YES);
    CHECK(state_of_xy(1, -1e-300) == D8L_YES_DISLIKE);
}

static void test_zero_and_outside_the_space(void) {
    CHECK(state_of_xy(0, 0) == D8L_INDIFFERENT);        /* indifference is exactly zero */
    CHECK(state_of_xy(-0.0, 0.0) == D8L_INDIFFERENT);
    CHECK(state_of_xy(NAN, 1) == D8L_UNDEFINED);        
    CHECK(state_of_xy(INFINITY, 1) == D8L_UNDEFINED);    /* non-finite input */
    CHECK(d8l_state_of(d8l_mul(d8l_nan(), canon(D8L_YES))) == D8L_UNDEFINED);
}

/* The decision is the sector: scale must never destroy it. */
static void test_overflow_keeps_the_decision(void) {
    d8l big = d8l_from_xy(1e200, 1e200);                 /* 45 degrees */
    CHECK(d8l_state_of(d8l_mul(big, big)) == D8L_LIKE);  /* 90 degrees, strength out of range */
    d8l huge = d8l_from_xy(1.7e308, 0);
    CHECK(d8l_state_of(d8l_add(huge, huge)) == D8L_YES);
}

static void test_underflow_is_not_indifference(void) {
    d8l tiny = d8l_from_xy(1e-200, 1e-200);
    CHECK(d8l_state_of(d8l_mul(tiny, d8l_from_xy(1e-200, 0))) == D8L_YES_LIKE);
    CHECK(d8l_state_of(d8l_mul(tiny, tiny)) == D8L_LIKE);
}

static int exact_sector(int re, int im) {                /* integer arithmetic, no floats */
    int q, u, v;
    if (re > 0 && im >= 0)       { q = 0; u = re;  v = im;  }
    else if (re <= 0 && im > 0)  { q = 1; u = im;  v = -re; }
    else if (re < 0 && im <= 0)  { q = 2; u = -re; v = -im; }
    else                         { q = 3; u = -im; v = re;  }
    return 2 * q + (v < u ? 0 : 1);
}

static void test_scaled_products_match_exact_arithmetic(void) {
    static const int scales[5] = {-1070, -600, 0, 600, 1020};   /* 2^-1070 .. 2^1020 */
    for (int a = -2; a <= 2; ++a) for (int b = -2; b <= 2; ++b) {
        if (a == 0 && b == 0) continue;
        for (int c = -2; c <= 2; ++c) for (int d = -2; d <= 2; ++d) {
            if (c == 0 && d == 0) continue;
            const int expected = exact_sector(a * c - b * d, a * d + b * c);
            for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) {
                d8l za = d8l_from_xy(ldexp(a, scales[i]), ldexp(b, scales[i]));
                d8l zb = d8l_from_xy(ldexp(c, scales[j]), ldexp(d, scales[j]));
                CHECK((int)d8l_state_of(d8l_mul(za, zb)) == expected);
            }
        }
    }
}

static void test_logic_axis_is_ternary(void) {
    const double vals[3] = {-2.0, 0.0, 3.0};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            d8l a = d8l_from_xy(vals[i], 0), b = d8l_from_xy(vals[j], 0);
            CHECK(d8l_state_of(d8l_mul(a, b)) == sign_state(vals[i] * vals[j]));
            CHECK(d8l_state_of(d8l_add(a, b)) == sign_state(vals[i] + vals[j]));
        }
}

static void test_composition(void) {
    d8l yes = canon(D8L_YES), like = canon(D8L_LIKE);
    d8l dislike = canon(D8L_DISLIKE), no = canon(D8L_NO), yl = canon(D8L_YES_LIKE);
    CHECK(d8l_state_of(d8l_mul(yes, like)) == D8L_LIKE);       /* YES is identity */
    CHECK(d8l_state_of(d8l_mul(like, like)) == D8L_NO);
    CHECK(d8l_state_of(d8l_mul(like, dislike)) == D8L_YES);
    CHECK(d8l_state_of(d8l_mul(no, no)) == D8L_YES);
    CHECK(d8l_state_of(d8l_mul(yl, yl)) == D8L_LIKE);          /* 45 + 45 = 90 */
}

static void test_absorbing_elements(void) {
    d8l x = canon(D8L_LIKE);
    CHECK(d8l_state_of(d8l_mul(d8l_zero(), x)) == D8L_INDIFFERENT);
    CHECK(d8l_state_of(d8l_mul(d8l_nan(), x)) == D8L_UNDEFINED);
    CHECK(d8l_state_of(d8l_mul(d8l_nan(), d8l_zero())) == D8L_UNDEFINED);
}

static void test_hidden_phase_inside_a_sector(void) {
    /* (3, 2) is in sector YES; composed with itself it is (5, 12): sector 1 */
    d8l a = d8l_from_xy(3, 2);
    CHECK(d8l_state_of(a) == D8L_YES);
    CHECK(d8l_state_of(d8l_mul(a, a)) == D8L_YES_LIKE);
}

static void test_magnitude_is_strength(void) {
    d8l a = d8l_from_state(D8L_NO, 0.5), b = d8l_from_state(D8L_LIKE, 4.0);
    CHECK(fabs(d8l_magnitude(a) - 0.5) < 1e-12);
    CHECK(fabs(d8l_magnitude(d8l_mul(a, b)) - 2.0) < 1e-12);
}

static void test_negations(void) {
    for (int j = 0; j < 8; ++j) {
        d8l z = canon((d8l_state)j);
        CHECK(d8l_state_of(d8l_neg(z)) == (d8l_state)((j + 4) % 8));
        CHECK(d8l_state_of(d8l_not_emotion(z)) == (d8l_state)((8 - j) % 8));
        CHECK(d8l_state_of(d8l_not_logic(z)) == (d8l_state)((4 - j + 8) % 8));
    }
}

static void test_superposition_and_to_xy(void) {
    CHECK(d8l_state_of(d8l_add(canon(D8L_YES), canon(D8L_LIKE))) == D8L_YES_LIKE);
    CHECK(d8l_state_of(d8l_add(canon(D8L_YES), canon(D8L_NO))) == D8L_INDIFFERENT);
    double l, e;
    d8l_to_xy(d8l_from_state(D8L_NO, 2.0), &l, &e);
    CHECK(l == -2.0 && e == 0.0);
    d8l_to_xy(d8l_from_state(D8L_DISLIKE, 2.0), &l, &e);
    CHECK(l == 0.0 && e == -2.0);
}

int main(void) {
    test_canonical_states_round_trip();
    test_half_open_sectors();
    test_zero_and_outside_the_space();
    test_overflow_keeps_the_decision();
    test_underflow_is_not_indifference();
    test_scaled_products_match_exact_arithmetic();
    test_logic_axis_is_ternary();
    test_composition();
    test_absorbing_elements();
    test_hidden_phase_inside_a_sector();
    test_magnitude_is_strength();
    test_negations();
    test_superposition_and_to_xy();
    printf("%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
