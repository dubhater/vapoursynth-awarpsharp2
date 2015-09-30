#include <cstdint>
#include <cstring>
#include <emmintrin.h>


#ifdef _WIN32
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif


static FORCE_INLINE void sobel_xmmword_u8_sse2(const uint8_t *srcp, uint8_t *dstp, int stride, __m128i th) {
    __m128i a11, a21, a31,
            a12,      a32,
            a13, a23, a33;

    a11 = _mm_loadu_si128((const __m128i *)(srcp - stride - 1));
    a21 = _mm_loadu_si128((const __m128i *)(srcp - stride));
    a31 = _mm_loadu_si128((const __m128i *)(srcp - stride + 1));

    a12 = _mm_loadu_si128((const __m128i *)(srcp - 1));
    a32 = _mm_loadu_si128((const __m128i *)(srcp + 1));

    a13 = _mm_loadu_si128((const __m128i *)(srcp + stride - 1));
    a23 = _mm_loadu_si128((const __m128i *)(srcp + stride));
    a33 = _mm_loadu_si128((const __m128i *)(srcp + stride + 1));

    __m128i avg_up    = _mm_avg_epu8(a21, _mm_avg_epu8(a11, a31));
    __m128i avg_down  = _mm_avg_epu8(a23, _mm_avg_epu8(a13, a33));
    __m128i avg_left  = _mm_avg_epu8(a12, _mm_avg_epu8(a13, a11));
    __m128i avg_right = _mm_avg_epu8(a32, _mm_avg_epu8(a33, a31));

    __m128i abs_v = _mm_or_si128(_mm_subs_epu8(avg_up, avg_down), _mm_subs_epu8(avg_down, avg_up));
    __m128i abs_h = _mm_or_si128(_mm_subs_epu8(avg_left, avg_right), _mm_subs_epu8(avg_right, avg_left));

    __m128i absolute = _mm_adds_epu8(abs_v, abs_h);

    __m128i abs_max = _mm_max_epu8(abs_h, abs_v);

    absolute = _mm_adds_epu8(absolute, abs_max);

    absolute = _mm_adds_epu8(_mm_adds_epu8(absolute, absolute), absolute);
    absolute = _mm_adds_epu8(absolute, absolute);

    _mm_storeu_si128((__m128i *)(dstp), _mm_min_epu8(absolute, th));
}


void sobel_u8_sse2(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh) {
    uint8_t *dstp_orig = dstp;

    srcp += stride;
    dstp += stride;

    __m128i th = _mm_set1_epi8(thresh);

    int width_sse2 = (width & ~15) + 2;
    if (width_sse2 > stride)
        width_sse2 -= 16;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width_sse2 - 1; x += 16)
            sobel_xmmword_u8_sse2(srcp + x, dstp + x, stride, th);

        if (width + 2 > width_sse2)
            sobel_xmmword_u8_sse2(srcp + width - 17, dstp + width - 17, stride, th);

        dstp[0] = dstp[1];
        dstp[width - 1] = dstp[width - 2];

        srcp += stride;
        dstp += stride;
    }

    memcpy(dstp_orig, dstp_orig + stride, width);
    memcpy(dstp, dstp - stride, width);
}
