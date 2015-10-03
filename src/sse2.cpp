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


static FORCE_INLINE void blur_r6_h_left_u8_sse2(const uint8_t *srcp, uint8_t *dstp) {
    __m128i avg12 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp + 1)), _mm_loadu_si128((const __m128i *)(srcp + 2)));
    __m128i avg34 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp + 3)), _mm_loadu_si128((const __m128i *)(srcp + 4)));
    __m128i avg56 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp + 5)), _mm_loadu_si128((const __m128i *)(srcp + 6)));

    __m128i avg012 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp)), avg12);
    __m128i avg3456 = _mm_avg_epu8(avg34, avg56);
    __m128i avg0123456 = _mm_avg_epu8(avg012, avg3456);
    __m128i avg = _mm_avg_epu8(avg012, avg0123456);

    _mm_storeu_si128((__m128i *)(dstp), avg);
}


static FORCE_INLINE void blur_r6_h_middle_u8_sse2(const uint8_t *srcp, uint8_t *dstp) {
    __m128i avg11 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 1)), _mm_loadu_si128((const __m128i *)(srcp + 1)));
    __m128i avg22 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 2)), _mm_loadu_si128((const __m128i *)(srcp + 2)));
    __m128i avg33 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 3)), _mm_loadu_si128((const __m128i *)(srcp + 3)));
    __m128i avg44 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 4)), _mm_loadu_si128((const __m128i *)(srcp + 4)));
    __m128i avg55 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 5)), _mm_loadu_si128((const __m128i *)(srcp + 5)));
    __m128i avg66 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 6)), _mm_loadu_si128((const __m128i *)(srcp + 6)));

    __m128i avg12 = _mm_avg_epu8(avg11, avg22);
    __m128i avg34 = _mm_avg_epu8(avg33, avg44);
    __m128i avg56 = _mm_avg_epu8(avg55, avg66);
    __m128i avg012 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp)), avg12);
    __m128i avg3456 = _mm_avg_epu8(avg34, avg56);
    __m128i avg0123456 = _mm_avg_epu8(avg012, avg3456);
    __m128i avg = _mm_avg_epu8(avg012, avg0123456);

    _mm_storeu_si128((__m128i *)(dstp), avg);
}


static FORCE_INLINE void blur_r6_h_right_u8_sse2(const uint8_t *srcp, uint8_t *dstp) {
    __m128i avg12 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 1)), _mm_loadu_si128((const __m128i *)(srcp - 2)));
    __m128i avg34 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 3)), _mm_loadu_si128((const __m128i *)(srcp - 4)));
    __m128i avg56 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 5)), _mm_loadu_si128((const __m128i *)(srcp - 6)));

    __m128i avg012 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp)), avg12);
    __m128i avg3456 = _mm_avg_epu8(avg34, avg56);
    __m128i avg0123456 = _mm_avg_epu8(avg012, avg3456);
    __m128i avg = _mm_avg_epu8(avg012, avg0123456);

    // This is the right edge. Only the highest six bytes are needed.
    int extra_bytes = *(int16_t *)(dstp + 8);
    avg = _mm_insert_epi16(avg, extra_bytes, 4);
    _mm_storeh_pi((__m64 *)(dstp + 8), _mm_castsi128_ps(avg));
}


static FORCE_INLINE void blur_r6_v_top_u8_sse2(const uint8_t *srcp, uint8_t *dstp, int stride) {
    __m128i l0 = _mm_loadu_si128((const __m128i *)(srcp));
    __m128i l1 = _mm_loadu_si128((const __m128i *)(srcp + stride));
    __m128i l2 = _mm_loadu_si128((const __m128i *)(srcp + stride * 2));
    __m128i l3 = _mm_loadu_si128((const __m128i *)(srcp + stride * 3));
    __m128i l4 = _mm_loadu_si128((const __m128i *)(srcp + stride * 4));
    __m128i l5 = _mm_loadu_si128((const __m128i *)(srcp + stride * 5));
    __m128i l6 = _mm_loadu_si128((const __m128i *)(srcp + stride * 6));

    __m128i avg12 = _mm_avg_epu8(l1, l2);
    __m128i avg34 = _mm_avg_epu8(l3, l4);
    __m128i avg56 = _mm_avg_epu8(l5, l6);

    __m128i avg3456 = _mm_avg_epu8(avg34, avg56);
    __m128i avg012 = _mm_avg_epu8(l0, avg12);
    __m128i avg0123456 = _mm_avg_epu8(avg012, avg3456);
    __m128i avg = _mm_avg_epu8(avg012, avg0123456);

    _mm_storeu_si128((__m128i *)(dstp), avg);
}


static FORCE_INLINE void blur_r6_v_middle_u8_sse2(const uint8_t *srcp, uint8_t *dstp, int stride) {
    __m128i m6 = _mm_loadu_si128((const __m128i *)(srcp - stride * 6));
    __m128i m5 = _mm_loadu_si128((const __m128i *)(srcp - stride * 5));
    __m128i m4 = _mm_loadu_si128((const __m128i *)(srcp - stride * 4));
    __m128i m3 = _mm_loadu_si128((const __m128i *)(srcp - stride * 3));
    __m128i m2 = _mm_loadu_si128((const __m128i *)(srcp - stride * 2));
    __m128i m1 = _mm_loadu_si128((const __m128i *)(srcp - stride));
    __m128i l0 = _mm_loadu_si128((const __m128i *)(srcp));
    __m128i l1 = _mm_loadu_si128((const __m128i *)(srcp + stride));
    __m128i l2 = _mm_loadu_si128((const __m128i *)(srcp + stride * 2));
    __m128i l3 = _mm_loadu_si128((const __m128i *)(srcp + stride * 3));
    __m128i l4 = _mm_loadu_si128((const __m128i *)(srcp + stride * 4));
    __m128i l5 = _mm_loadu_si128((const __m128i *)(srcp + stride * 5));
    __m128i l6 = _mm_loadu_si128((const __m128i *)(srcp + stride * 6));

    __m128i avg11 = _mm_avg_epu8(m1, l1);
    __m128i avg22 = _mm_avg_epu8(m2, l2);
    __m128i avg33 = _mm_avg_epu8(m3, l3);
    __m128i avg44 = _mm_avg_epu8(m4, l4);
    __m128i avg55 = _mm_avg_epu8(m5, l5);
    __m128i avg66 = _mm_avg_epu8(m6, l6);

    __m128i avg12 = _mm_avg_epu8(avg11, avg22);
    __m128i avg34 = _mm_avg_epu8(avg33, avg44);
    __m128i avg56 = _mm_avg_epu8(avg55, avg66);
    __m128i avg012 = _mm_avg_epu8(l0, avg12);
    __m128i avg3456 = _mm_avg_epu8(avg34, avg56);
    __m128i avg0123456 = _mm_avg_epu8(avg012, avg3456);
    __m128i avg = _mm_avg_epu8(avg012, avg0123456);

    _mm_storeu_si128((__m128i *)(dstp), avg);
}


static FORCE_INLINE void blur_r6_v_bottom_u8_sse2(const uint8_t *srcp, uint8_t *dstp, int stride) {
    __m128i m6 = _mm_loadu_si128((const __m128i *)(srcp - stride * 6));
    __m128i m5 = _mm_loadu_si128((const __m128i *)(srcp - stride * 5));
    __m128i m4 = _mm_loadu_si128((const __m128i *)(srcp - stride * 4));
    __m128i m3 = _mm_loadu_si128((const __m128i *)(srcp - stride * 3));
    __m128i m2 = _mm_loadu_si128((const __m128i *)(srcp - stride * 2));
    __m128i m1 = _mm_loadu_si128((const __m128i *)(srcp - stride));
    __m128i l0 = _mm_loadu_si128((const __m128i *)(srcp));

    __m128i avg12 = _mm_avg_epu8(m1, m2);
    __m128i avg34 = _mm_avg_epu8(m3, m4);
    __m128i avg56 = _mm_avg_epu8(m5, m6);
    __m128i avg012 = _mm_avg_epu8(l0, avg12);
    __m128i avg3456 = _mm_avg_epu8(avg34, avg56);
    __m128i avg0123456 = _mm_avg_epu8(avg012, avg3456);
    __m128i avg = _mm_avg_epu8(avg012, avg0123456);

    _mm_storeu_si128((__m128i *)(dstp), avg);
}


void blur_r6_u8_sse2(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    // Horizontal blur from mask to temp.
    // Vertical blur from temp back to mask.

    int width_sse2 = (width & ~15) + 12;
    if (width_sse2 > stride)
        width_sse2 -= 16;

    uint8_t *mask_orig = mask;
    uint8_t *temp_orig = temp;

    // Horizontal blur.

    for (int y = 0; y < height; y++) {
        blur_r6_h_left_u8_sse2(mask, temp);

        for (int x = 6; x < width_sse2 - 6; x += 16)
            blur_r6_h_middle_u8_sse2(mask + x, temp + x);

        if (width + 12 > width_sse2)
            blur_r6_h_middle_u8_sse2(mask + width - 22, temp + width - 22);

        blur_r6_h_right_u8_sse2(mask + width - 16, temp + width - 16);

        mask += stride;
        temp += stride;
    }


    // Vertical blur.

    width_sse2 = width & ~15;

    mask = mask_orig;
    temp = temp_orig;
    int y;

    for (y = 0; y < 6; y++) {
        for (int x = 0; x < width_sse2; x += 16)
            blur_r6_v_top_u8_sse2(temp + x, mask + x, stride);

        if (width > width_sse2)
            blur_r6_v_top_u8_sse2(temp + width - 16, mask + width - 16, stride);

        mask += stride;
        temp += stride;
    }

    for ( ; y < height - 6; y++) {
        for (int x = 0; x < width_sse2; x += 16)
            blur_r6_v_middle_u8_sse2(temp + x, mask + x, stride);

        if (width > width_sse2)
            blur_r6_v_middle_u8_sse2(temp + width - 16, mask + width - 16, stride);

        mask += stride;
        temp += stride;
    }

    for ( ; y < height; y++) {
        for (int x = 0; x < width_sse2; x += 16)
            blur_r6_v_bottom_u8_sse2(temp + x, mask + x, stride);

        if (width > width_sse2)
            blur_r6_v_bottom_u8_sse2(temp + width - 16, mask + width - 16, stride);

        mask += stride;
        temp += stride;
    }
}


static FORCE_INLINE void blur_r2_h_u8_sse2(const uint8_t *srcp, uint8_t *dstp) {
    __m128i avg1 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 1)), _mm_loadu_si128((const __m128i *)(srcp + 1)));
    __m128i avg2 = _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(srcp - 2)), _mm_loadu_si128((const __m128i *)(srcp + 2)));
    __m128i middle = _mm_loadu_si128((const __m128i *)(srcp));
    __m128i avg = _mm_avg_epu8(avg2, middle);
    avg = _mm_avg_epu8(avg, middle);
    avg = _mm_avg_epu8(avg, avg1);

    _mm_storeu_si128((__m128i *)(dstp), avg);
}


static FORCE_INLINE void blur_r2_v_u8_sse2(const uint8_t *srcp, uint8_t *dstp, int stride_p2, int stride_p1, int stride_n1, int stride_n2) {
    __m128i m2 = _mm_loadu_si128((const __m128i *)(srcp + stride_p2));
    __m128i m1 = _mm_loadu_si128((const __m128i *)(srcp + stride_p1));
    __m128i l0 = _mm_loadu_si128((const __m128i *)(srcp));
    __m128i l1 = _mm_loadu_si128((const __m128i *)(srcp + stride_n1));
    __m128i l2 = _mm_loadu_si128((const __m128i *)(srcp + stride_n2));

    __m128i avg1 = _mm_avg_epu8(m1, l1);
    __m128i avg2 = _mm_avg_epu8(m2, l2);
    __m128i avg = _mm_avg_epu8(avg2, l0);
    avg = _mm_avg_epu8(avg, l0);
    avg = _mm_avg_epu8(avg, avg1);

    _mm_storeu_si128((__m128i *)(dstp), avg);
}


void blur_r2_u8_sse2(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    // Horizontal blur from mask to temp.
    // Vertical blur from temp back to mask.

    int width_sse2 = (width & ~15) + 4;
    if (width_sse2 > stride)
        width_sse2 -= 16;

    uint8_t *mask_orig = mask;
    uint8_t *temp_orig = temp;

    // Horizontal blur.

    for (int y = 0; y < height; y++) {
        int avg, avg1, avg2;

        avg1 = (mask[0] + mask[1] + 1) >> 1;
        avg2 = (mask[0] + mask[2] + 1) >> 1;
        avg = (avg2 + mask[0] + 1) >> 1;
        avg = (avg + mask[0] + 1) >> 1;
        avg = (avg + avg1 + 1) >> 1;

        temp[0] = avg;

        avg1 = (mask[0] + mask[2] + 1) >> 1;
        avg2 = (mask[0] + mask[3] + 1) >> 1;
        avg = (avg2 + mask[1] + 1) >> 1;
        avg = (avg + mask[1] + 1) >> 1;
        avg = (avg + avg1 + 1) >> 1;

        temp[1] = avg;

        for (int x = 2; x < width_sse2 - 2; x += 16)
            blur_r2_h_u8_sse2(mask + x, temp + x);

        if (width + 4 > width_sse2)
            blur_r2_h_u8_sse2(mask + width - 18, temp + width - 18);

        avg1 = (mask[width - 3] + mask[width - 1] + 1) >> 1;
        avg2 = (mask[width - 4] + mask[width - 1] + 1) >> 1;
        avg = (avg2 + mask[width - 2] + 1) >> 1;
        avg = (avg + mask[width - 2] + 1) >> 1;
        avg = (avg + avg1 + 1) >> 1;

        temp[width - 2] = avg;

        avg1 = (mask[width - 2] + mask[width - 1] + 1) >> 1;
        avg2 = (mask[width - 3] + mask[width - 1] + 1) >> 1;
        avg = (avg2 + mask[width - 1] + 1) >> 1;
        avg = (avg + mask[width - 1] + 1) >> 1;
        avg = (avg + avg1 + 1) >> 1;

        temp[width - 1] = avg;

        mask += stride;
        temp += stride;
    }


    // Vertical blur.

    width_sse2 = width & ~15;

    mask = mask_orig;
    temp = temp_orig;

    for (int y = 0; y < height; y++) {
        int stride_p1 = y ? -stride : 0;
        int stride_p2 = y > 1 ? stride_p1 * 2 : stride_p1;
        int stride_n1 = y < height - 1 ? stride : 0;
        int stride_n2 = y < height - 2 ? stride_n1 * 2 : stride_n1;

        for (int x = 0; x < width_sse2; x += 16)
            blur_r2_v_u8_sse2(temp + x, mask + x, stride_p2, stride_p1, stride_n1, stride_n2);

        if (width > width_sse2)
            blur_r2_v_u8_sse2(temp + width - 16, mask + width - 16, stride_p2, stride_p1, stride_n1, stride_n2);

        mask += stride;
        temp += stride;
    }
}
