#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

// D8 complex value: coordinates live exclusively in the closed wedge
// 0 <= y <= x.  The orientation is an 8-bit sector mask.
// Interior: one bit; axis/diagonal: two adjacent bits;
// zero: 0xFF; NaN state: 0x00.
//
// Multiplication is branch-first, entirely local, and uses no trig and
// no global Re/Im conversion.  See README.md for the mathematical rules.

struct D8Complex final {
    double x = 0.0;
    double y = 0.0;
    std::uint8_t rotation = 0xFF;

    static constexpr std::uint8_t rol(std::uint8_t m, unsigned n = 1) noexcept {
        n &= 7u;
        return static_cast<std::uint8_t>((static_cast<unsigned>(m) << n) |
                                         (static_cast<unsigned>(m) >> ((8u - n) & 7u)));
    }
    static constexpr std::uint8_t ror(std::uint8_t m, unsigned n = 1) noexcept {
        return rol(m, (8u - (n & 7u)) & 7u);
    }

    static constexpr D8Complex zero() noexcept { return {0.0, 0.0, 0xFF}; }
    static constexpr D8Complex nan() noexcept { return {0.0, 0.0, 0x00}; }

    constexpr bool is_zero() const noexcept { return rotation == 0xFF; }
    constexpr bool is_nan() const noexcept { return rotation == 0x00; }

    static constexpr int bit_index(std::uint8_t m) noexcept {
        // Only called on a one-hot mask; comparison chain is portable C++17.
        int k = 0;
        while ((m >>= 1u) != 0u) ++k;
        return k;
    }

    static constexpr int even_anchor(std::uint8_t m) noexcept {
        // Every canonical two-bit boundary has a unique even-position bit.
        if ((m & static_cast<std::uint8_t>(m - 1u)) == 0u) return bit_index(m);
        for (int k = 0; k < 8; k += 2) {
            if (m & static_cast<std::uint8_t>(1u << k)) return k;
        }
        return -1;
    }

    static constexpr bool is_one_hot(std::uint8_t m) noexcept {
        return m != 0 && (m & static_cast<std::uint8_t>(m - 1u)) == 0;
    }

    static D8Complex make(double x, double y, std::uint8_t mask) {
        if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0)
            throw std::invalid_argument("x and y must be finite and nonnegative");
        if (mask == 0x00) {
            if (x != 0 || y != 0) throw std::invalid_argument("NaN mask requires (0,0)");
            return nan();
        }
        if (x == 0 && y == 0) return zero();
        if (mask == 0xFF) throw std::invalid_argument("0xFF is reserved for zero");

        if (is_one_hot(mask)) {
            int k = bit_index(mask);
            if (y > x) { std::swap(x, y); k ^= 1; }
            return reduced(x, y, static_cast<std::uint8_t>(1u << k));
        }

        int bit_count = 0;
        for (int k = 0; k < 8; ++k) bit_count += (mask >> k) & 1u;
        if (bit_count != 2 || y > x || (y != 0 && x != y))
            throw std::invalid_argument("invalid boundary representation");
        const int k = even_anchor(mask);
        const int neighbor = (y == 0) ? ((k + 7) & 7) : ((k + 1) & 7);
        const auto expected = static_cast<std::uint8_t>((1u << k) | (1u << neighbor));
        if (mask != expected) throw std::invalid_argument("boundary mask incompatible with coordinates");
        return {x, y, mask};
    }

    // This function is used for an already normalized *one-bit* working counter.
    // It adds a second bit only when the final point lies exactly on a boundary.
    static constexpr D8Complex reduced(double x, double y, std::uint8_t counter) noexcept {
        if (x == 0.0 && y == 0.0) return zero();
        const int k = bit_index(counter);
        if (y == 0.0) {
            const int neighbor = (k & 1) ? ((k + 1) & 7) : ((k + 7) & 7);
            counter = static_cast<std::uint8_t>(counter | (1u << neighbor));
        } else if (x == y) {
            const int neighbor = (k & 1) ? ((k + 7) & 7) : ((k + 1) & 7);
            counter = static_cast<std::uint8_t>(counter | (1u << neighbor));
        }
        return {x, y, counter};
    }

    D8Complex multiply(const D8Complex& other) const noexcept {
        if (is_nan() || other.is_nan()) return nan();
        if (is_zero() || other.is_zero()) return zero();

        const int ka = even_anchor(rotation);
        const int kb = even_anchor(other.rotation);
        const int qa = ((ka + 1) / 2) & 3;
        const int qb = ((kb + 1) / 2) & 3;
        const int pa = ka & 1;
        const int pb = kb & 1;

        const double ac = x * other.x;
        const double bd = y * other.y;
        const double ad = x * other.y;
        const double bc = y * other.x;

        // Start from a one-hot counter. A two-bit boundary is never rotated.
        std::uint8_t counter = rol(0x01, static_cast<unsigned>(2 * ((qa + qb) & 3)));
        double out_x, out_y;

        if (pa == pb) {
            // (ad + bc) > (ac - bd)  <=>  ad + bc + bd > ac
            const bool cross_diagonal = ad + bc + bd > ac;
            if (pa == 0) {
                if (cross_diagonal) counter = rol(counter);
            } else {
                counter = ror(counter);
                if (cross_diagonal) counter = ror(counter);
            }
            if (cross_diagonal) {
                out_x = ad + bc;
                out_y = ac - bd;
            } else {
                out_x = ac - bd;
                out_y = ad + bc;
            }
        } else {
            const bool negative_deviation = (pa == 0) ? (bc < ad) : (ad < bc);
            if (negative_deviation) counter = ror(counter);
            out_x = ac + bd;
            out_y = (bc >= ad) ? (bc - ad) : (ad - bc);
        }
        // If floating products overflow, return the explicit NaN state.
        if (!std::isfinite(out_x) || !std::isfinite(out_y)) return nan();
        return reduced(out_x, out_y, counter);
    }

    D8Complex operator*(const D8Complex& rhs) const noexcept { return multiply(rhs); }
};
