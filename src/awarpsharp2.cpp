#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <algorithm>

#include <VapourSynth.h>
#include <VSHelper.h>


#if defined(AWARPSHARP2_X86)
extern void sobel_u8_sse2(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh);
extern void blur_r6_u8_sse2(uint8_t *mask, uint8_t *temp, int stride, int width, int height);
extern void blur_r2_u8_sse2(uint8_t *mask, uint8_t *temp, int stride, int width, int height);
extern void warp0_u8_sse2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int stride, int edge_stride, int width, int height, int depth);
extern void warp2_u8_sse2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int stride, int edge_stride, int width, int height, int depth);
#endif


static void sobel_c(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh) {
    uint8_t *dstp_orig = dstp;

    srcp += stride;
    dstp += stride;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int a11, a21, a31,
                a12,      a32,
                a13, a23, a33;

            a11 = srcp[x - stride - 1];
            a21 = srcp[x - stride];
            a31 = srcp[x - stride + 1];

            a12 = srcp[x - 1];
            a32 = srcp[x + 1];

            a13 = srcp[x + stride - 1];
            a23 = srcp[x + stride];
            a33 = srcp[x + stride + 1];

            int avg_up    = (a21 + ((a11 + a31 + 1) >> 1) + 1) >> 1;
            int avg_down  = (a23 + ((a13 + a33 + 1) >> 1) + 1) >> 1;
            int avg_left  = (a12 + ((a13 + a11 + 1) >> 1) + 1) >> 1;
            int avg_right = (a32 + ((a33 + a31 + 1) >> 1) + 1) >> 1;

            int abs_v = abs(avg_up - avg_down);
            int abs_h = abs(avg_left - avg_right);

            int absolute = std::min(abs_v + abs_h, 255);

            int abs_max = std::max(abs_h, abs_v);

            absolute = std::min(absolute + abs_max, 255);

            absolute = std::min(std::min(absolute * 2, 255) + absolute, 255);
            absolute = std::min(absolute * 2, 255);

            dstp[x] = std::min(absolute, thresh);
        }

        dstp[0] = dstp[1];
        dstp[width - 1] = dstp[width - 2];

        srcp += stride;
        dstp += stride;
    }

    memcpy(dstp_orig, dstp_orig + stride, width);
    memcpy(dstp, dstp - stride, width);
}


static void blur_r6_c(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    // Horizontal blur from mask to temp.
    // Vertical blur from temp back to mask.

    uint8_t *mask_orig = mask;
    uint8_t *temp_orig = temp;

    // Horizontal blur.

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < 6; x++) {
            int avg12 = (mask[x + 1] + mask[x + 2] + 1) >> 1;
            int avg34 = (mask[x + 3] + mask[x + 4] + 1) >> 1;
            int avg56 = (mask[x + 5] + mask[x + 6] + 1) >> 1;

            int avg012 = (mask[x] + avg12 + 1) >> 1;
            int avg3456 = (avg34 + avg56 + 1) >> 1;
            int avg0123456 = (avg012 + avg3456 + 1) >> 1;
            int avg = (avg012 + avg0123456 + 1) >> 1;

            temp[x] = avg;
        }

        for (int x = 6; x < width - 6; x++) {
            int avg11 = (mask[x - 1] + mask[x + 1] + 1) >> 1;
            int avg22 = (mask[x - 2] + mask[x + 2] + 1) >> 1;
            int avg33 = (mask[x - 3] + mask[x + 3] + 1) >> 1;
            int avg44 = (mask[x - 4] + mask[x + 4] + 1) >> 1;
            int avg55 = (mask[x - 5] + mask[x + 5] + 1) >> 1;
            int avg66 = (mask[x - 6] + mask[x + 6] + 1) >> 1;

            int avg12 = (avg11 + avg22 + 1) >> 1;
            int avg34 = (avg33 + avg44 + 1) >> 1;
            int avg56 = (avg55 + avg66 + 1) >> 1;
            int avg012 = (mask[x] + avg12 + 1) >> 1;
            int avg3456 = (avg34 + avg56 + 1) >> 1;
            int avg0123456 = (avg012 + avg3456 + 1) >> 1;
            int avg = (avg012 + avg0123456 + 1) >> 1;

            temp[x] = avg;
        }

        for (int x = width - 6; x < width; x++) {
            int avg12 = (mask[x - 1] + mask[x - 2] + 1) >> 1;
            int avg34 = (mask[x - 3] + mask[x - 4] + 1) >> 1;
            int avg56 = (mask[x - 5] + mask[x - 6] + 1) >> 1;

            int avg012 = (mask[x] + avg12 + 1) >> 1;
            int avg3456 = (avg34 + avg56 + 1) >> 1;
            int avg0123456 = (avg012 + avg3456 + 1) >> 1;
            int avg = (avg012 + avg0123456 + 1) >> 1;

            temp[x] = avg;
        }

        mask += stride;
        temp += stride;
    }


    // Vertical blur.
    
    mask = mask_orig;
    temp = temp_orig;
    int y;

    for (y = 0; y < 6; y++) {
        for (int x = 0; x < width; x++) {
            int l0 = temp[x];
            int l1 = temp[x + stride];
            int l2 = temp[x + stride * 2];
            int l3 = temp[x + stride * 3];
            int l4 = temp[x + stride * 4];
            int l5 = temp[x + stride * 5];
            int l6 = temp[x + stride * 6];

            int avg12 = (l1 + l2 + 1) >> 1;
            int avg34 = (l3 + l4 + 1) >> 1;
            int avg56 = (l5 + l6 + 1) >> 1;

            int avg3456 = (avg34 + avg56 + 1) >> 1;
            int avg012 = (l0 + avg12 + 1) >> 1;
            int avg0123456 = (avg012 + avg3456 + 1) >> 1;
            int avg = (avg012 + avg0123456 + 1) >> 1;

            mask[x] = avg;
        }

        mask += stride;
        temp += stride;
    }

    for ( ; y < height - 6; y++) {
        for (int x = 0; x < width; x++) {
            int m6 = temp[x - stride * 6];
            int m5 = temp[x - stride * 5];
            int m4 = temp[x - stride * 4];
            int m3 = temp[x - stride * 3];
            int m2 = temp[x - stride * 2];
            int m1 = temp[x - stride];
            int l0 = temp[x];
            int l1 = temp[x + stride];
            int l2 = temp[x + stride * 2];
            int l3 = temp[x + stride * 3];
            int l4 = temp[x + stride * 4];
            int l5 = temp[x + stride * 5];
            int l6 = temp[x + stride * 6];

            int avg11 = (m1 + l1 + 1) >> 1;
            int avg22 = (m2 + l2 + 1) >> 1;
            int avg33 = (m3 + l3 + 1) >> 1;
            int avg44 = (m4 + l4 + 1) >> 1;
            int avg55 = (m5 + l5 + 1) >> 1;
            int avg66 = (m6 + l6 + 1) >> 1;

            int avg12 = (avg11 + avg22 + 1) >> 1;
            int avg34 = (avg33 + avg44 + 1) >> 1;
            int avg56 = (avg55 + avg66 + 1) >> 1;
            int avg012 = (l0 + avg12 + 1) >> 1;
            int avg3456 = (avg34 + avg56 + 1) >> 1;
            int avg0123456 = (avg012 + avg3456 + 1) >> 1;
            int avg = (avg012 + avg0123456 + 1) >> 1;

            mask[x] = avg;
        }

        mask += stride;
        temp += stride;
    }

    for ( ; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int m6 = temp[x - stride * 6];
            int m5 = temp[x - stride * 5];
            int m4 = temp[x - stride * 4];
            int m3 = temp[x - stride * 3];
            int m2 = temp[x - stride * 2];
            int m1 = temp[x - stride];
            int l0 = temp[x];

            int avg12 = (m1 + m2 + 1) >> 1;
            int avg34 = (m3 + m4 + 1) >> 1;
            int avg56 = (m5 + m6 + 1) >> 1;
            int avg012 = (l0 + avg12 + 1) >> 1;
            int avg3456 = (avg34 + avg56 + 1) >> 1;
            int avg0123456 = (avg012 + avg3456 + 1) >> 1;
            int avg = (avg012 + avg0123456 + 1) >> 1;

            mask[x] = avg;
        }

        mask += stride;
        temp += stride;
    }
}


static void blur_r2_c(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    // Horizontal blur from mask to temp.
    // Vertical blur from temp back to mask.

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

        for (int x = 2; x < width - 2; x++) {
            avg1 = (mask[x - 1] + mask[x + 1] + 1) >> 1;
            avg2 = (mask[x - 2] + mask[x + 2] + 1) >> 1;
            avg = (avg2 + mask[x] + 1) >> 1;
            avg = (avg + mask[x] + 1) >> 1;
            avg = (avg + avg1 + 1) >> 1;

            temp[x] = avg;
        }

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


    // Vertical blur

    mask = mask_orig;
    temp = temp_orig;

    for (int y = 0; y < height; y++) {
        int stride_p1 = y ? -stride : 0;
        int stride_p2 = y > 1 ? stride_p1 * 2 : stride_p1;
        int stride_n1 = y < height - 1 ? stride : 0;
        int stride_n2 = y < height - 2 ? stride_n1 * 2 : stride_n1;

        for (int x = 0; x < width; x++) {
            int m2 = temp[x + stride_p2];
            int m1 = temp[x + stride_p1];
            int l0 = temp[x];
            int l1 = temp[x + stride_n1];
            int l2 = temp[x + stride_n2];

            int avg1 = (m1 + l1 + 1) >> 1;
            int avg2 = (m2 + l2 + 1) >> 1;
            int avg = (avg2 + l0 + 1) >> 1;
            avg = (avg + l0 + 1) >> 1;
            avg = (avg + avg1 + 1) >> 1;

            mask[x] = avg;
        }

        mask += stride;
        temp += stride;
    }
}


static void bilinear_downscale_h_c(uint8_t *srcp, int src_stride, int src_width, int src_height) {
    for (int y = 0; y < src_height; y++) {
        for (int x = 0; x < src_width / 2; x++)
            srcp[x] = (srcp[x * 2] + srcp[x * 2 + 1] + 1) >> 1;

        srcp += src_stride;
    }
}


static void bilinear_downscale_v_c(uint8_t *srcp, int src_stride, int src_width, int src_height) {
    for (int y = 0; y < src_height / 2; y++) {
        for (int x = 0; x < src_width; x++)
            srcp[x] = (srcp[x + y * src_stride] + srcp[x + (y + 1) * src_stride] + 1) >> 1;

        srcp += src_stride;
    }
}


static void bilinear_downscale_hv_c(uint8_t *srcp, int src_stride, int src_width, int src_height) {
    for (int y = 0; y < src_height / 2; y++) {
        for (int x = 0; x < src_width / 2; x++) {
            int avg1 = (srcp[x * 2 + y * src_stride] + srcp[x * 2 + 1 + y * src_stride] + 1) >> 1;
            int avg2 = (srcp[x * 2 + (y + 1) * src_stride] + srcp[x * 2 + 1 + (y + 1) * src_stride] + 1) >> 1;
            srcp[x] = (avg1 + avg2 + 1) >> 1;
        }

        srcp += src_stride;
    }
}


template <int SMAGL> // 0 or 2
static void warp_c(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int stride, int edge_stride, int width, int height, int depth) {
    int SMAG = 1 << SMAGL;

    // Because the width of the source is width of the destination times SMAG.
    int src_stride = stride * SMAG;

    depth <<= 8;

    const int x_limit_min = 0 * SMAG;
    const int x_limit_max = (width - 1) * SMAG;

    for (int y = 0; y < height; y++) {
        int y_limit_min = -y * 128;
        int y_limit_max = (height - y) * 128 - 129; // (height - y - 1) * 128 - 1

        for (int x = 0; x < width; x++) {
            // calculate displacement

            int above, below;
            if (y == 0)
                above = edgep[x];
            else
                above = edgep[-edge_stride + x];

            if (y == height - 1)
                below = edgep[x];
            else
                below = edgep[edge_stride + x];

            int left, right;
            if (x == 0)
                left = edgep[x];
            else
                left = edgep[x - 1];

            if (x == width - 1)
                right = edgep[x];
            else
                right = edgep[x + 1];

            int h = left - right;
            int v = above - below;

            h <<= 7;
            v <<= 7;

            h *= depth;
            h >>= 16;
            v *= depth;
            v >>= 16;

            v = std::max(v, y_limit_min);
            v = std::min(v, y_limit_max);

            int remainder_h = h;
            int remainder_v = v;

            if (SMAGL) {
                remainder_h <<= SMAGL; // shift by 2; multiply by 4
                remainder_v <<= SMAGL;
            }

            remainder_h &= 127; // remainder of the division by 128 (or 32 if it was shifted left by 2 above)
            remainder_v &= 127;

            h >>= 7 - SMAGL; // shift by 7 (or 5); division by 128 (or 32)
            v >>= 7 - SMAGL;

            h += x << SMAGL;
            h = std::min(std::max(h, -32768), 32767); // likely pointless

            bool remainder_needed = (x_limit_max > h) && !(x_limit_min > h);
            if (!remainder_needed)
                remainder_h = 0; // probably correct

            h = std::min(h, x_limit_max);
            h = std::max(h, x_limit_min);

            // h and v contain the displacement now.

            int s00 = srcp[v * src_stride + h];
            int s01 = srcp[v * src_stride + h + 1];
            int s10 = srcp[(v + 1) * src_stride + h];
            int s11 = srcp[(v + 1) * src_stride + h + 1];

            int s0 = s00 * (128 - remainder_h);
            int s1 = s10 * (128 - remainder_h);

            s0 += s01 * remainder_h;
            s1 += s11 * remainder_h;

            s0 += 64;
            s1 += 64;

            s0 >>= 7;
            s1 >>= 7;

            s0 *= 128 - remainder_v;
            s1 *= remainder_v;

            int s = s0 + s1;

            s += 64;

            s >>= 7;

            dstp[x] = std::min(std::max(s, 0), 255);
        }

        srcp += src_stride * SMAG;
        edgep += edge_stride;
        dstp += stride;
    }
}


typedef struct {
    VSNodeRef *node;
    const VSVideoInfo *vi;

    int thresh;
    int blur_level;
    int blur_type;
    int depth;
    int chroma;
    int process[3];
    int opt;

    void (*edge_mask)(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh);
    void (*blur)(uint8_t *mask, uint8_t *temp, int stride, int width, int height);
    void (*bilinear_downscale)(uint8_t *srcp, int src_stride, int src_width, int src_height);
    void (*warp)(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int stride, int edge_stride, int width, int height, int depth);
} AWarpSharp2Data;


static void VS_CC aWarpSharp2Init(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    AWarpSharp2Data *d = (AWarpSharp2Data *) * instanceData;
    vsapi->setVideoInfo(d->vi, 1, node);
}


static const VSFrameRef *VS_CC aWarpSharp2GetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    const AWarpSharp2Data *d = (const AWarpSharp2Data *) * instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef *frames[3] = {
            d->process[0] ? nullptr : src,
            d->process[1] ? nullptr : src,
            d->process[2] ? nullptr : src
        };
        int planes[3] = { 0, 1, 2 };

        const VSFormat *fmt = vsapi->getFrameFormat(src);
        int width_y = vsapi->getFrameWidth(src, 0);
        int height_y = vsapi->getFrameHeight(src, 0);

        VSFrameRef *dst = vsapi->newVideoFrame2(fmt, width_y, height_y, frames, planes, src, core);

        uint8_t *mask_y = nullptr;

        if (d->process[0] || ((d->process[1] || d->process[2]) && d->chroma == 0 && fmt->numPlanes > 1)) {
            int stride_y = vsapi->getStride(src, 0);

            mask_y = vs_aligned_malloc<uint8_t>(height_y * stride_y, 32);

            const uint8_t *srcp = vsapi->getReadPtr(src, 0);
            uint8_t *dstp = vsapi->getWritePtr(dst, 0);

            d->edge_mask(srcp, mask_y, stride_y, width_y, height_y, d->thresh);

            for (int i = 0; i < d->blur_level; i++)
                d->blur(mask_y, dstp, stride_y, width_y, height_y);

            if (d->process[0])
                d->warp(srcp, mask_y, dstp, stride_y, stride_y, width_y, height_y, d->depth);
            else
                vs_bitblt(dstp, stride_y, srcp, stride_y, width_y, height_y);
        }

        if ((d->process[1] || d->process[2]) && fmt->numPlanes > 1) {
            if (d->chroma == 1) {
                int stride_uv = vsapi->getStride(src, 1);
                int width_uv = vsapi->getFrameWidth(src, 1);
                int height_uv = vsapi->getFrameHeight(src, 1);

                uint8_t *mask_uv = vs_aligned_malloc<uint8_t>(height_uv * stride_uv, 32);

                for (int plane = 1; plane < fmt->numPlanes; plane++) {
                    if (!d->process[plane])
                        continue;

                    const uint8_t *srcp = vsapi->getReadPtr(src, plane);
                    uint8_t *dstp = vsapi->getWritePtr(dst, plane);

                    d->edge_mask(srcp, mask_uv, stride_uv, width_uv, height_uv, d->thresh);

                    for (int i = 0; i < (d->blur_level + 1) / 2; i++)
                        d->blur(mask_uv, dstp, stride_uv, width_uv, height_uv);

                    d->warp(srcp, mask_uv, dstp, stride_uv, stride_uv, width_uv, height_uv, d->depth / 2);
                }

                vs_aligned_free(mask_uv);
            } else if (d->chroma == 0) {
                int stride_y = vsapi->getStride(src, 0);
                int stride_uv = vsapi->getStride(src, 1);
                int width_uv = vsapi->getFrameWidth(src, 1);
                int height_uv = vsapi->getFrameHeight(src, 1);

                if (d->bilinear_downscale)
                    d->bilinear_downscale(mask_y, stride_y, d->vi->width, d->vi->height);

                for (int plane = 1; plane < fmt->numPlanes; plane++) {
                    if (!d->process[plane])
                        continue;

                    const uint8_t *srcp = vsapi->getReadPtr(src, plane);
                    uint8_t *dstp = vsapi->getWritePtr(dst, plane);

                    d->warp(srcp, mask_y, dstp, stride_uv, stride_y, width_uv, height_uv, d->depth / 2);
                }
            }
        }


        if (mask_y)
            vs_aligned_free(mask_y);

        vsapi->freeFrame(src);

        return dst;
    }

    return 0;
}


static void VS_CC aWarpSharp2Free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    AWarpSharp2Data *d = (AWarpSharp2Data *)instanceData;

    vsapi->freeNode(d->node);
    free(d);
}


static void selectFunctions(AWarpSharp2Data *d) {
    d->edge_mask = sobel_c;

    if (d->blur_type == 0)
        d->blur = blur_r6_c;
    else
        d->blur = blur_r2_c;

    if (d->vi->format->subSamplingW && d->vi->format->subSamplingH)
        d->bilinear_downscale = bilinear_downscale_hv_c;
    else if (d->vi->format->subSamplingW)
        d->bilinear_downscale = bilinear_downscale_h_c;
    else if (d->vi->format->subSamplingH)
        d->bilinear_downscale = bilinear_downscale_v_c;
    else
        d->bilinear_downscale = nullptr;

    d->warp = warp_c<0>;

#if defined(AWARPSHARP2_X86)
    if (d->opt) {
        d->edge_mask = sobel_u8_sse2;

        if (d->blur_type == 0)
            d->blur = blur_r6_u8_sse2;
        else
            d->blur = blur_r2_u8_sse2;

        d->warp = warp0_u8_sse2;
    }
#endif
}


static void VS_CC aWarpSharp2Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    AWarpSharp2Data d;
    AWarpSharp2Data *data;

    memset(&d, 0, sizeof(d));

    int err;

    d.thresh = int64ToIntS(vsapi->propGetInt(in, "thresh", 0, &err));
    if (err)
        d.thresh = 128;

    d.blur_type = int64ToIntS(vsapi->propGetInt(in, "type", 0, &err));

    d.blur_level = int64ToIntS(vsapi->propGetInt(in, "blur", 0, &err));
    if (err)
        d.blur_level = d.blur_type ? 3 : 2;

    d.depth = int64ToIntS(vsapi->propGetInt(in, "depth", 0, &err));
    if (err)
        d.depth = 16;

    d.chroma = int64ToIntS(vsapi->propGetInt(in, "chroma", 0, &err));

    d.opt = !!vsapi->propGetInt(in, "opt", 0, &err);
    if (err)
        d.opt = 1;


    if (d.thresh < 0 || d.thresh > 255) {
        vsapi->setError(out, "AWarpSharp2: thresh must be between 0 and 255 (inclusive).");
        return;
    }

    if (d.blur_level < 0) {
        vsapi->setError(out, "AWarpSharp2: blur must be at least 0.");
        return;
    }

    if (d.blur_type < 0 || d.blur_type > 1) {
        vsapi->setError(out, "AWarpSharp2: type must be 0 or 1.");
        return;
    }

    if (d.depth < -128 || d.depth > 127) {
        vsapi->setError(out, "AWarpSharp2: depth must be between -128 and 127 (inclusive).");
        return;
    }

    if (d.chroma < 0 || d.chroma > 1) {
        vsapi->setError(out, "AWarpSharp2: chroma must be 0 or 1.");
        return;
    }


    d.node = vsapi->propGetNode(in, "clip", 0, NULL);
    d.vi = vsapi->getVideoInfo(d.node);


    if (!d.vi->format || d.vi->format->bitsPerSample > 8 || d.vi->format->colorFamily == cmRGB) {
        vsapi->setError(out, "AWarpSharp2: only constant format, 8 bit, not RGB clips supported.");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi->format->subSamplingW > 1 || d.vi->format->subSamplingH > 1) {
        vsapi->setError(out, "AWarpSharp2: the chroma subsampling ratio cannot be greater than 2.");
        vsapi->freeNode(d.node);
        return;
    }

    int n = d.vi->format->numPlanes;
    int m = vsapi->propNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        d.process[i] = (m <= 0);

    for (int i = 0; i < m; i++) {
        int o = int64ToIntS(vsapi->propGetInt(in, "planes", i, 0));

        if (o < 0 || o >= n) {
            vsapi->setError(out, "AWarpSharp2: plane index out of range.");
            vsapi->freeNode(d.node);
            return;
        }

        if (d.process[o]) {
            vsapi->setError(out, "AWarpSharp2: plane specified twice.");
            vsapi->freeNode(d.node);
            return;
        }

        d.process[o] = 1;
    }


    selectFunctions(&d);


    data = (AWarpSharp2Data *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "AWarpSharp2", aWarpSharp2Init, aWarpSharp2GetFrame, aWarpSharp2Free, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.nodame.awarpsharp2", "warp", "Sharpen images by warping", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("AWarpSharp2",
            "clip:clip;"
            "thresh:int:opt;"
            "blur:int:opt;"
            "type:int:opt;"
            "depth:int:opt;"
            "chroma:int:opt;"
            "planes:int[]:opt;"
            "opt:int:opt;"
            , aWarpSharp2Create, 0, plugin);
}
