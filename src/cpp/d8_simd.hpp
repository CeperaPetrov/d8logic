#pragma once
#include "d8_complex.hpp"
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

// SoA representation: x[i]>=y[i]>=0 for all finite nonzero elements.
// m: one-hot interior; two adjacent bits on boundary; FF zero; 00 NaN.
struct D8SoA {
    const double* ax; const double* ay; const std::uint8_t* am;
    const double* bx; const double* by; const std::uint8_t* bm;
    double* ox; double* oy; std::uint8_t* om;
};

static inline int d8_anchor(std::uint8_t mask) noexcept {
    unsigned m = mask;
    unsigned one = (m & (m-1u)) ? (m & 0x55u) : m;
#if defined(_MSC_VER)
    unsigned long k;
    _BitScanForward(&k, static_cast<unsigned long>(one));
    return int(k);
#else
    return __builtin_ctz(one);
#endif
}

static inline std::uint8_t d8_encode_mask(double x, double y, int k) noexcept {
    if (x == 0.0 && y == 0.0) return 0xff;
    // The working counter is ALWAYS one-hot. Rotate that bit, never a
    // two-bit boundary mask. A boundary adds one neighbour at the very end.
    const unsigned idx = unsigned(k) & 7u;
    const std::uint8_t counter = D8Complex::rol(0x01, idx);
    if (y == 0.0)
        return static_cast<std::uint8_t>(counter | ((idx & 1u) ? D8Complex::rol(counter) : D8Complex::ror(counter)));
    if (x == y)
        return static_cast<std::uint8_t>(counter | ((idx & 1u) ? D8Complex::ror(counter) : D8Complex::rol(counter)));
    return counter;
}


// Enumerate the D8 orbit without repeating the multiplication.
// The one-hot working counter is rotated; a 2-bit boundary is synthesized
// afresh for each orientation and NEVER rotated as a pair.
static inline void d8_orbit_masks(const D8Complex& value, std::uint8_t masks[8]) noexcept {
    if (value.is_zero() || value.is_nan()) {
        for (int i=0;i<8;++i) masks[i]=value.rotation;
        return;
    }
    const int anchor=d8_anchor(value.rotation);
    for(int i=0;i<8;++i)
        masks[i]=d8_encode_mask(value.x,value.y,(anchor+i)&7);
}

static inline void d8_scalar_one(
    double a, double b, unsigned ma, double c, double d, unsigned mb,
    double& x, double& y, std::uint8_t& m) noexcept {
    if (!ma || !mb) { x = 0; y = 0; m = 0; return; }
    if (ma == 0xff || mb == 0xff) { x = 0; y = 0; m = 0xff; return; }
    const int ka = d8_anchor(static_cast<std::uint8_t>(ma));
    const int kb = d8_anchor(static_cast<std::uint8_t>(mb));
    const int pa = ka & 1, pb = kb & 1;
    const int k0 = 2 * ((((ka+1)/2)+((kb+1)/2)) & 3);
    // All four products are calculated regardless of sector.
    const double ac=a*c, bd=b*d, ad=a*d, bc=b*c;
    const bool same = (((ka ^ kb) & 1) == 0);
    const double diff = ac-bd, sum = ad+bc;
    const bool cross = (sum+bd > ac);
    const bool greater = (bc >= ad);
    if (same) {
        x = cross ? sum : diff;
        y = cross ? diff : sum;
    } else {
        x = ac+bd;
        y = greater ? bc-ad : ad-bc;
    }
    const int k = same ? (k0 + (pa ? (-1-int(cross)) : int(cross)))
                       : (k0 - int(pa ? greater : !greater));
    if (!std::isfinite(x) || !std::isfinite(y)) { x=0; y=0; m=0; return; }
    m = d8_encode_mask(x,y,k);
}

static inline void d8_scalar_array(const D8SoA& z, std::size_t n) noexcept {
    for (std::size_t i=0;i<n;++i)
        d8_scalar_one(z.ax[i],z.ay[i],z.am[i],z.bx[i],z.by[i],z.bm[i],
                      z.ox[i],z.oy[i],z.om[i]);
}

#if defined(__AVX2__)
// A full 8-sector AVX2 kernel. 4 independent pairs at once. The floating
// arithmetic is vectorized. The masks and ring counter are handled per lane.
// No global cartesian Re/Im conversion, trig, or signed local coordinates.
static inline void d8_avx2_array(const D8SoA& z, std::size_t n) noexcept {
    std::size_t i=0;
    alignas(32) double tmpx[4], tmpy[4];
    for (;i+4<=n;i+=4) {
        const __m256d a=_mm256_loadu_pd(z.ax+i), b=_mm256_loadu_pd(z.ay+i);
        const __m256d c=_mm256_loadu_pd(z.bx+i), d=_mm256_loadu_pd(z.by+i);
        const __m256d ac=_mm256_mul_pd(a,c), bd=_mm256_mul_pd(b,d);
        const __m256d ad=_mm256_mul_pd(a,d), bc=_mm256_mul_pd(b,c);
        const __m256d diff=_mm256_sub_pd(ac,bd);
        const __m256d sum=_mm256_add_pd(ad,bc);
        const __m256d cross=_mm256_cmp_pd(_mm256_add_pd(sum,bd),ac,_CMP_GT_OQ);
        const __m256d greater=_mm256_cmp_pd(bc,ad,_CMP_GE_OQ);
        const __m256d same_x=_mm256_blendv_pd(diff,sum,cross);
        const __m256d same_y=_mm256_blendv_pd(sum,diff,cross);
        const __m256d mixed_x=_mm256_add_pd(ac,bd);
        const __m256d mixed_y=_mm256_blendv_pd(_mm256_sub_pd(ad,bc),_mm256_sub_pd(bc,ad),greater);
        unsigned same_bits=0;
        int k0[4],pa[4];
        for (int lane=0;lane<4;++lane) {
            const int ka=d8_anchor(z.am[i+lane] == 0 || z.am[i+lane] == 255 ? std::uint8_t(1) : z.am[i+lane]);
            const int kb=d8_anchor(z.bm[i+lane] == 0 || z.bm[i+lane] == 255 ? std::uint8_t(1) : z.bm[i+lane]);
            pa[lane]=ka&1;
            same_bits |= unsigned(((ka ^ kb) & 1)==0) << lane;
            k0[lane]=2*((((ka+1)/2)+((kb+1)/2))&3);
        }
        const __m256i predicate=_mm256_set_epi64x(
             -std::int64_t((same_bits>>3)&1),-std::int64_t((same_bits>>2)&1),
             -std::int64_t((same_bits>>1)&1),-std::int64_t(same_bits&1));
        const __m256d same_mask=_mm256_castsi256_pd(predicate);
        const __m256d vx=_mm256_blendv_pd(mixed_x,same_x,same_mask);
        const __m256d vy=_mm256_blendv_pd(mixed_y,same_y,same_mask);
        _mm256_storeu_pd(tmpx,vx);
        _mm256_storeu_pd(tmpy,vy);
        const int cross_bits=_mm256_movemask_pd(cross);
        const int greater_bits=_mm256_movemask_pd(greater);
        for(int lane=0;lane<4;++lane) {
            unsigned ma=z.am[i+lane], mb=z.bm[i+lane];
            if (!ma || !mb) { z.ox[i+lane]=0;z.oy[i+lane]=0;z.om[i+lane]=0;continue; }
            if (ma==255 || mb==255) {z.ox[i+lane]=0;z.oy[i+lane]=0;z.om[i+lane]=255;continue;}
            const bool same=(same_bits>>lane)&1;
            const bool cross_diag=(cross_bits>>lane)&1;
            const bool greater_ab=(greater_bits>>lane)&1;
            const int k = same ? (k0[lane]+(pa[lane] ? (-1-int(cross_diag)) : int(cross_diag)))
                               : (k0[lane]-int(pa[lane] ? greater_ab : !greater_ab));
            double x=tmpx[lane], y=tmpy[lane];
            if (!std::isfinite(x)||!std::isfinite(y)) { x=0;y=0;z.om[i+lane]=0; }
            else z.om[i+lane]=d8_encode_mask(x,y,k);
            z.ox[i+lane]=x;z.oy[i+lane]=y;
        }
    }
    for(;i<n;++i)
        d8_scalar_one(z.ax[i],z.ay[i],z.am[i],z.bx[i],z.by[i],z.bm[i],
                      z.ox[i],z.oy[i],z.om[i]);
}
#endif
