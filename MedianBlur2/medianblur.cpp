#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "avisynth.h"
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <emmintrin.h>
#include "special.h"

enum class InstructionSet
{
    SSE2,
    PLAIN_C
};

static __forceinline int calculate_window_side_length(int radius, int x, int width) {
    int length = radius + 1;
    if (x <= radius) {
        length += x;
    } else if (x >= width-radius) {
        length += width-x-1;
    } else {
        length += radius;
    }
    return length;
}

template<typename T>
static __forceinline __m128i simd_adds(const __m128i &a, const __m128i &b) {
    assert(false);
}

template<>
static __forceinline __m128i simd_adds<uint8_t>(const __m128i &a, const __m128i &b) {
    return _mm_adds_epu8(a, b);
}

template<>
static __forceinline __m128i simd_adds<uint16_t>(const __m128i &a, const __m128i &b) {
    return _mm_adds_epu16(a, b);
}

template<>
static __forceinline __m128i simd_adds<int32_t>(const __m128i &a, const __m128i &b) {
    return _mm_add_epi32(a, b);
}

template<typename T>
static __forceinline __m128i simd_subs(const __m128i &a, const __m128i &b) {
    assert(false);
}

template<>
static __forceinline __m128i simd_subs<uint8_t>(const __m128i &a, const __m128i &b) {
    return _mm_subs_epu8(a, b);
}

template<>
static __forceinline __m128i simd_subs<uint16_t>(const __m128i &a, const __m128i &b) {
    return _mm_subs_epu16(a, b);
}

template<>
static __forceinline __m128i simd_subs<int32_t>(const __m128i &a, const __m128i &b) {
    return _mm_sub_epi32(a, b);
}


template<typename T, InstructionSet instruction_set>
class MedianProcessor
{
    typedef struct {
        T coarse[16];
        T fine[256];
    } Histogram;

    typedef struct {
        int start;
        int end;
    } ColumnPair;

    static __forceinline void add_16_bins_c(T* a, const T *b) {
        for (int i = 0; i < 16; ++i) {
            a[i] += b[i];
        }
    }

    static __forceinline void sub_16_bins_c(T* a, const T *b) {
        for (int i = 0; i < 16; ++i) {
            a[i] -= b[i];
        }
    }

    static __forceinline void zero_single_bin_c(T* a) {
        for (int i = 0; i < 16; ++i) {
            a[i] = 0;
        }
    }

    static __forceinline void add_16_bins_sse2(T* a, const T *b) {
        for (int i = 0; i < sizeof(T); ++i) {
            __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a)+i);
            __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b)+i);
            __m128i sum = simd_adds<T>(aval, bval);
            _mm_store_si128(reinterpret_cast<__m128i*>(a)+i, sum);
        }
    }

    static __forceinline void sub_16_bins_sse2(T* a, const T *b) {
        for (int i = 0; i < sizeof(T); ++i) {
            __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a)+i);
            __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b)+i);
            __m128i sum = simd_subs<T>(aval, bval);
            _mm_store_si128(reinterpret_cast<__m128i*>(a)+i, sum);
        }
    }

    static __forceinline void zero_single_bin_sse2(T* a) {
        __m128i zero = _mm_setzero_si128();
        for (int i = 0; i < sizeof(T); ++i) {
            _mm_store_si128(reinterpret_cast<__m128i*>(a)+i, zero);
        }
    }

    static __forceinline void add_16_bins(T* a, const T *b) {
        if (instruction_set == InstructionSet::SSE2) {
            add_16_bins_sse2(a, b);
        } else {
            add_16_bins_c(a, b);
        }
    }

    static __forceinline void sub_16_bins(T* a, const T *b) {
        if (instruction_set == InstructionSet::SSE2) {
            sub_16_bins_sse2(a, b);
        } else {
            sub_16_bins_c(a, b);
        }
    }

    static __forceinline void zero_single_bin(T* a) {
        if (instruction_set == InstructionSet::SSE2) {
            zero_single_bin_sse2(a);
        } else {
            zero_single_bin_c(a);
        }
    }

    static __forceinline void process_line(uint8_t *dstp, Histogram* histograms, ColumnPair fine_columns[16], Histogram &current_hist, int radius, int temporal_radius, int width, int current_length_y) {
        memset(current_hist.coarse, 0, HISTOGRAM_SIZE);
        memset(fine_columns, -1, sizeof(ColumnPair) * 16);
        //init histogram of the leftmost column
        for (int x = 0; x < radius; ++x) {
            add_16_bins(current_hist.coarse, histograms[x].coarse);
        }

        for (int x = 0; x < width; ++x) {
            //add one column to the right
            if (x < (width-radius)) {
                add_16_bins(current_hist.coarse, histograms[x+radius].coarse);
            }

            int current_length_x = calculate_window_side_length(radius, x, width);
            int start_x = std::max(0, x-radius);
            int end_x = start_x + current_length_x;
            int half_elements = (current_length_y * current_length_x * temporal_radius + 1) / 2;

            //finding median on coarse level
            int count = 0;
            int coarse_idx = -1;
            while (count < half_elements) {
                count += current_hist.coarse[++coarse_idx];
            }
            count -= current_hist.coarse[coarse_idx];

            assert(coarse_idx < 16);
            int fine_offset = coarse_idx*16;

            //partially updating fine histogram
            if (fine_columns[coarse_idx].end < start_x) {
                //any data we gathered is useless, drop it and build the whole block from scratch
                zero_single_bin(current_hist.fine + fine_offset);

                for (int i = start_x; i < end_x; ++i) {
                    add_16_bins(current_hist.fine+fine_offset, histograms[i].fine+fine_offset);
                }
            } else {
                int i = fine_columns[coarse_idx].start;
                while (i < start_x) {
                    sub_16_bins(current_hist.fine+fine_offset, histograms[i++].fine+fine_offset);
                }
                i = fine_columns[coarse_idx].end;
                while (++i < end_x) {
                    add_16_bins(current_hist.fine+fine_offset, histograms[i].fine+fine_offset);
                }
            }
            fine_columns[coarse_idx].start = start_x;
            fine_columns[coarse_idx].end = end_x-1;

            //finding median on fine level
            int fine_idx = fine_offset - 1;
            while (count < half_elements) {
                count += current_hist.fine[++fine_idx];
            }

            assert(fine_idx < 256);
            dstp[x] = fine_idx;

            //subtract leftmost histogram
            if (x >= radius) {
                sub_16_bins(current_hist.coarse, histograms[x-radius].coarse);
            }
        }
    }

public:
    static const int HISTOGRAM_SIZE = sizeof(Histogram);

    static void calculate_median(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void *buffer) {
        Histogram* histograms = reinterpret_cast<Histogram*>(buffer);
        __declspec(align(16)) Histogram current_hist;
        ColumnPair fine_columns[16]; //indexes of last and first column in every fine histogram segment
        memset(histograms, 0, width * HISTOGRAM_SIZE);

        //init histograms
        for (int y = 0; y < radius+1; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t new_element = srcp[y*src_pitch+x];
                histograms[x].coarse[new_element>>4]++;
                histograms[x].fine[new_element]++;
            }
        }

        for (int y = 0; y < height; ++y) {
            int current_length_y = calculate_window_side_length(radius, y, height);

            process_line(dstp, histograms, fine_columns, current_hist, radius, 1, width, current_length_y);

            //updating column histograms
            if (y >= radius) {
                for (int x = 0; x < width; ++x) {
                    uint8_t old_element = srcp[(y-radius)*src_pitch+x];
                    histograms[x].coarse[old_element>>4]--;
                    histograms[x].fine[old_element]--;
                }
            }
            if (y < height-radius-1) {
                for (int x = 0; x < width; ++x) {
                    uint8_t new_element = srcp[(y+radius+1)*src_pitch+x];
                    histograms[x].coarse[new_element>>4]++;
                    histograms[x].fine[new_element]++;
                }
            }

            dstp += dst_pitch;
        }
    }


    static void calculate_temporal_median(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void *buffer) {
        Histogram* histograms = reinterpret_cast<Histogram*>(buffer);
        __declspec(align(16)) Histogram current_hist;
        ColumnPair fine_columns[16]; //indexes of last and first column in every fine histogram segment
        memset(histograms, 0, width * HISTOGRAM_SIZE);

        //init histograms
        for (int y = 0; y < radius+1; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int i = 0; i < frames_count; ++i) {
                    uint8_t new_element = src_ptrs[i][y*src_pitches[i]+x];
                    histograms[x].coarse[new_element>>4]++;
                    histograms[x].fine[new_element]++;
                }
            }
        }

        for (int y = 0; y < height; ++y) {
            int current_length_y = calculate_window_side_length(radius, y, height);

            process_line(dstp, histograms, fine_columns, current_hist, radius, frames_count, width, current_length_y);

            //updating column histograms
            if (y >= radius) {
                for (int x = 0; x < width; ++x) {
                    for (int i = 0; i < frames_count; ++i) {
                        uint8_t old_element = src_ptrs[i][(y-radius)*src_pitches[i]+x];
                        histograms[x].coarse[old_element>>4]--;
                        histograms[x].fine[old_element]--;
                    }
                }
            }
            if (y < height-radius-1) {
                for (int x = 0; x < width; ++x) {
                    for (int i = 0; i < frames_count; ++i) {
                        uint8_t new_element = src_ptrs[i][(y+radius+1)*src_pitches[i]+x];
                        histograms[x].coarse[new_element>>4]++;
                        histograms[x].fine[new_element]++;
                    }
                }
            }

            dstp += dst_pitch;
        }
    }
};


class MedianBlur : public GenericVideoFilter {
public:
    MedianBlur(PClip child, int radius_y, int radius_u, int radius_v, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);

    ~MedianBlur() {
        _aligned_free(buffer_);
    }

private:
    int radius_y_;
    int radius_u_;
    int radius_v_;
    void *buffer_;
    decltype(&MedianProcessor<uint8_t, InstructionSet::SSE2>::calculate_median) processors_[3];

    static const int MAX_RADIUS = 127;
    static const int MIN_MODE = -255;
};

MedianBlur::MedianBlur(PClip child, int radius_y, int radius_u, int radius_v, IScriptEnvironment* env)
: GenericVideoFilter(child), radius_y_(radius_y), radius_u_(radius_u), radius_v_(radius_v), buffer_(nullptr) {
    if (!vi.IsPlanar()) {
        env->ThrowError("MedianBlur: only planar formats allowed");
    }

    if (radius_y > MAX_RADIUS || radius_u > MAX_RADIUS || radius_v > MAX_RADIUS) {
        env->ThrowError("MedianBlur: radius is too large. Must be between 0 and %i", MAX_RADIUS);
    }
    if (radius_y < MIN_MODE || radius_u < MIN_MODE || radius_v < MIN_MODE) {
        env->ThrowError("MedianBlur: radius is too small. Must be between 0 and %i", MAX_RADIUS);
    }

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int radii[] = { radius_y, radius_u, radius_v };
    int min_width = vi.width;

#pragma warning(disable: 4800)
    bool sse2 = env->GetCPUFlags() & CPUF_SSE2;
#pragma warning(default: 4800)

    int hist_size = 0;
    for (int i = 0; i < (vi.IsY8() ? 1 : 3); ++i) {
        if (radii[i] <= 0) {
            continue;
        }
        int width = vi.width >> vi.GetPlaneWidthSubsampling(planes[i]);
        int height = vi.width >> vi.GetPlaneHeightSubsampling(planes[i]);
        int core_size = radii[i]*2 + 1;
        if (width < core_size || height < core_size) {
            env->ThrowError("MedianBlur: image is too small for this radius!");
        }

        //special cases make sense only when SSE2 is available, otherwise generic routine will be faster
        if (radii[i] == 1 && sse2 && width > 16) {
            processors_[i] = &calculate_median_r1;
        } else if (radii[i] == 2 && sse2 && width > 16) {
            processors_[i] = &calculate_median_r2;
        } else if (radii[i] < 8) {
            hist_size = std::max(hist_size, MedianProcessor<uint8_t, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE);
            if (sse2) {
                processors_[i] = &MedianProcessor<uint8_t, InstructionSet::SSE2>::calculate_median;
            } else {
                processors_[i] = &MedianProcessor<uint8_t, InstructionSet::PLAIN_C>::calculate_median;
            }
        } else {
            hist_size = std::max(hist_size, MedianProcessor<uint16_t, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE);
            if (sse2) {
                processors_[i] = &MedianProcessor<uint16_t, InstructionSet::SSE2>::calculate_median;
            } else {
                processors_[i] = &MedianProcessor<uint16_t, InstructionSet::PLAIN_C>::calculate_median;
            }
        }
    }

    if (!sse2 || radius_u > 2 || radius_v > 2 || radius_y > 2) {
        //allocate buffer only for generic approach
        buffer_ = _aligned_malloc(vi.width  * hist_size, 16);
        if (!buffer_) {
            env->ThrowError("MedianBlurTemp: Couldn't callocate buffer.");
        }
    }
}

PVideoFrame MedianBlur::GetFrame(int n, IScriptEnvironment *env) {
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int radii[] = { radius_y_, radius_u_, radius_v_ };
    for (int i = 0; i < (vi.IsY8() ? 1 : 3); i++) {
        int plane = planes[i];
        int radius = radii[i];
        int width = src->GetRowSize(plane);
        int height = src->GetHeight(plane);

        if (radius > 0) {
            processors_[i](dst->GetWritePtr(plane), src->GetReadPtr(plane), dst->GetPitch(plane),
                src->GetPitch(plane), width, height, radius, buffer_);
        } else if (radius == 0) {
            env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane),
                src->GetReadPtr(plane), src->GetPitch(plane), width, height);
        } else {
            memset(dst->GetWritePtr(plane), -radius, dst->GetPitch(plane)*height);
        }
    }

    return dst;
}



class MedianBlurTemp : public GenericVideoFilter {
public:
    MedianBlurTemp(PClip child, int radius_y, int radius_u, int radius_v, int radius_temp, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);

    ~MedianBlurTemp() {
        _aligned_free(buffer_);
    }

private:
    int radius_y_;
    int radius_u_;
    int radius_v_;
    int radius_temp_;
    void *buffer_;
    decltype(&MedianProcessor<int32_t, InstructionSet::SSE2>::calculate_temporal_median) processor_;

    static const int MAX_RADIUS = 1024;
    static const int MIN_MODE = -255;
};

MedianBlurTemp::MedianBlurTemp(PClip child, int radius_y, int radius_u, int radius_v, int radius_temp, IScriptEnvironment* env)
: GenericVideoFilter(child), radius_y_(radius_y), radius_u_(radius_u), radius_v_(radius_v), radius_temp_(radius_temp), buffer_(nullptr) {
    if (!vi.IsPlanar()) {
        env->ThrowError("MedianBlurTemp: only planar formats allowed");
    }

    if (radius_y > MAX_RADIUS || radius_u > MAX_RADIUS || radius_v > MAX_RADIUS) {
        env->ThrowError("MedianBlurTemp: radius is too large. Must be between 0 and %i", MAX_RADIUS);
    }
    if (radius_y < MIN_MODE || radius_u < MIN_MODE || radius_v < MIN_MODE) {
        env->ThrowError("MedianBlurTemp: radius is too small. Must be between 0 and %i", MAX_RADIUS);
    }
    if (radius_temp <= 0) {
        env->ThrowError("MedianBlurTemp: Invalid temporal radius. Should be greater than zero.");
    }

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int radii[] = { radius_y, radius_u, radius_v };
    int min_width = vi.width;

    for (int i = 0; i < (vi.IsY8() ? 1 : 3); ++i) {
        if (radii[i] < 0) {
            continue;
        }
        int width = vi.width >> vi.GetPlaneWidthSubsampling(planes[i]);
        int height = vi.width >> vi.GetPlaneHeightSubsampling(planes[i]);
        int core_size = radii[i]*2 + 1;
        if (width < core_size || height < core_size) {
            env->ThrowError("MedianBlurTemp: image is too small for this radius!");
        }
    }

    if (env->GetCPUFlags() & CPUF_SSE2) {
        processor_ = &MedianProcessor<int32_t, InstructionSet::SSE2>::calculate_temporal_median;
    } else {
        processor_ = &MedianProcessor<int32_t, InstructionSet::PLAIN_C>::calculate_temporal_median;
    }

    buffer_ = _aligned_malloc(vi.width  * MedianProcessor<int32_t, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE, 16);
    if (!buffer_) {
        env->ThrowError("MedianBlurTemp: Couldn't callocate buffer.");
    }
}

PVideoFrame MedianBlurTemp::GetFrame(int n, IScriptEnvironment *env) {
    int frame_buffer_size = sizeof(PVideoFrame)* (radius_temp_*2+1);
    PVideoFrame* src_frames = reinterpret_cast<PVideoFrame*>(alloca(frame_buffer_size));
    if (src_frames == nullptr) {
        env->ThrowError("MedianBlurTemp: Couldn't allocate memory on stack. This is a bug, please report");
    }
    memset(src_frames, 0, frame_buffer_size);

    PVideoFrame src = child->GetFrame(n, env);
    int frame_count = 0;
    for (int i = -radius_temp_; i <= radius_temp_; ++i) {
        int frame_number = n + i;
        //don't get the n'th frame twice
        src_frames[frame_count++] = (i == 0 ? src : child->GetFrame(frame_number, env));
    }
    const uint8_t **src_ptrs = reinterpret_cast<const uint8_t **>(alloca(sizeof(uint8_t*)* frame_count));
    int *src_pitches = reinterpret_cast<int*>(alloca(sizeof(int)* frame_count));
    if (src_ptrs == nullptr || src_pitches == nullptr) {
        env->ThrowError("MedianBlurTemp: Couldn't allocate memory on stack. This is a bug, please report");
    }

    PVideoFrame dst = env->NewVideoFrame(vi);

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int radii[] = { radius_y_, radius_u_, radius_v_ };
    for (int i = 0; i < (vi.IsY8() ? 1 : 3); i++) {
        int plane = planes[i];
        int radius = radii[i];
        int width = dst->GetRowSize(plane);
        int height = dst->GetHeight(plane);

        for (int i = 0; i < frame_count; ++i) {
            src_ptrs[i] = src_frames[i]->GetReadPtr(plane);
            src_pitches[i] = src_frames[i]->GetPitch(plane);
        }

        if (radius >= 0) {
            processor_(dst->GetWritePtr(plane), dst->GetPitch(plane), 
                src_ptrs, src_pitches, frame_count, width, height, radius, buffer_);
        } else if (radius == -1) {
            env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane),
                src->GetReadPtr(plane), src->GetPitch(plane), width, height);
        } else {
            memset(dst->GetWritePtr(plane), -radius, dst->GetPitch(plane)*height);
        }
    }

    for (int i = 0; i < frame_count; ++i) {
        src_frames[i].~PVideoFrame();
    }

    return dst;
}

AVSValue __cdecl create_median_blur(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, RADIUS, RADIUS_U, RADIUS_V };
    return new MedianBlur(args[CLIP].AsClip(), args[RADIUS].AsInt(2), args[RADIUS_U].AsInt(2), args[RADIUS_V].AsInt(2), env);
}

AVSValue __cdecl create_temporal_median_blur(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, RADIUS, RADIUS_U, RADIUS_V, TEMPORAL_RADIUS };
    return new MedianBlurTemp(args[CLIP].AsClip(), args[RADIUS].AsInt(2), args[RADIUS_U].AsInt(2), args[RADIUS_V].AsInt(2), args[TEMPORAL_RADIUS].AsInt(1), env);
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    env->AddFunction("MedianBlur", "c[radiusy]i[radiusu]i[radiusv]i", create_median_blur, 0);
    env->AddFunction("MedianBlurTemporal", "c[radiusy]i[radiusu]i[radiusv]i[temporalradius]i", create_temporal_median_blur, 0);
    return "Kawaikunai";
}