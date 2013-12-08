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
            __m128i sum = _mm_adds_epu16(aval, bval);
            _mm_store_si128(reinterpret_cast<__m128i*>(a)+i, sum);
        }
    }

    static __forceinline void sub_16_bins_sse2(T* a, const T *b) {
        for (int i = 0; i < sizeof(T); ++i) {
            __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a)+i);
            __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b)+i);
            __m128i sum = _mm_subs_epu16(aval, bval);
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

            memset(current_hist.coarse, 0, HISTOGRAM_SIZE);
            memset(fine_columns, -1, sizeof(fine_columns));
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
                int half_elements = (current_length_y * current_length_x + 1) / 2;

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
    decltype(&MedianProcessor<uint8_t, InstructionSet::SSE2>::calculate_median) processor_;

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
    for (int i = 0; i < (vi.IsY8() ? 1 : 3); ++i) {
        if (radii[i] <= 0) {
            continue;
        }
        int width = vi.width >> vi.GetPlaneWidthSubsampling(planes[i]);
        int height = vi.width >> vi.GetPlaneHeightSubsampling(planes[i]);
        int core_size = radii[i]*2 + 1;
        min_width = std::min(width, min_width);
        if (width < core_size || height < core_size) {
            env->ThrowError("MedianBlur: image is too small for this radius!");
        }
    }

#pragma warning(disable: 4800)
    bool sse2 = env->GetCPUFlags() & CPUF_SSE2;
#pragma warning(default: 4800)

    int max_radius = std::max(std::max(radius_y, radius_v), radius_v);
    if (max_radius > 0) {
        int hist_size;
        //special cases make sense only when SSE2 is available, otherwise generic routine will be faster
        if (max_radius == 1 && sse2 && min_width > 16) {
            processor_ = &calculate_median_r1;
        } else if (max_radius == 2 && sse2 && min_width > 16) {
            processor_ = &calculate_median_r2;
        } else if (max_radius < 8) {
            hist_size = MedianProcessor<uint8_t, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE;
            if (sse2) {
                processor_ = &MedianProcessor<uint8_t, InstructionSet::SSE2>::calculate_median;
            } else {
                processor_ = &MedianProcessor<uint8_t, InstructionSet::PLAIN_C>::calculate_median;
            }
        } else {
            hist_size = MedianProcessor<uint16_t, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE;
            if (sse2) {
                processor_ = &MedianProcessor<uint16_t, InstructionSet::SSE2>::calculate_median;
            } else {
                processor_ = &MedianProcessor<uint16_t, InstructionSet::PLAIN_C>::calculate_median;
            }
        }

        if (!sse2 || max_radius > 2) {
            //allocate buffer only for generic approach
            buffer_ = _aligned_malloc((vi.width + radius_y * 2) * hist_size, 16);
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
            processor_(dst->GetWritePtr(plane), src->GetReadPtr(plane), dst->GetPitch(plane),
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

AVSValue __cdecl create_median_blur(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, RADIUS, RADIUS_U, RADIUS_V };
    return new MedianBlur(args[CLIP].AsClip(), args[RADIUS].AsInt(2), args[RADIUS_U].AsInt(2), args[RADIUS_V].AsInt(2), env);
}


const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    env->AddFunction("MedianBlur", "c[radiusy]i[radiusu]i[radiusv]i", create_median_blur, 0);
    return "Kawaikunai";
}