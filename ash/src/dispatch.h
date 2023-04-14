// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>

#include "minivec.h"

#ifndef _MSC_VER
#define UNUSED_ATTRIBUTE __attribute__((unused))
#else
#define UNUSED_ATTRIBUTE
#endif

#ifdef __CUDACC__
#define FN_SPECIFIERS __host__ __device__
#else
#define FN_SPECIFIERS
#endif

#define INSTANTIATE_TYPES(DTYPE, DIM)       \
    using key_t = MiniVec<DTYPE, DIM>;      \
    using hash_t = MiniVecHash<DTYPE, DIM>; \
    using eq_t = MiniVecEq<DTYPE, DIM>;

const int kMaxKeyDim = 4;

#define DIM_SWITCHER(DTYPE, DIM, ...)                                \
    if (DIM == 1) {                                                  \
        INSTANTIATE_TYPES(DTYPE, 1)                                  \
        return __VA_ARGS__();                                        \
    } else if (DIM == 2) {                                           \
        INSTANTIATE_TYPES(DTYPE, 2)                                  \
        return __VA_ARGS__();                                        \
    } else if (DIM == 3) {                                           \
        INSTANTIATE_TYPES(DTYPE, 3)                                  \
        return __VA_ARGS__();                                        \
    } else if (DIM == 4) {                                           \
        INSTANTIATE_TYPES(DTYPE, 4)                                  \
        return __VA_ARGS__();                                        \
    } else {                                                         \
        std::cerr << "Unsupported dim " << DIM << ", please modify " \
                  << __FILE__ << " and compile from source\n";       \
    }

// TODO: dispatch more combinations.
#define DISPATCH_DTYPE_AND_DIM_TO_TEMPLATE(DTYPE, DIM, ...)    \
    [&] {                                                      \
        if (DTYPE == torch::kInt32) {                          \
            DIM_SWITCHER(int, DIM, __VA_ARGS__)                \
        } else {                                               \
            std::cerr << "Unsupported dtype " << DTYPE         \
                      << ", please use integer types (Int64, " \
                         "Int32, Int16).\n";                   \
        }                                                      \
    }()

#ifdef __CUDACC__
// Reinterpret hash maps' void* value arrays as CUDA primitive types arrays, to
// avoid slow memcpy or byte-by-byte copy in kernels.
// Not used in the CPU version since memcpy is relatively fast on CPU.
#define DISPATCH_DIVISOR_SIZE_TO_BLOB_T(DIVISOR, ...) \
    [&] {                                             \
        if (DIVISOR == 16) {                          \
            using blob_t = int4;                      \
            return __VA_ARGS__();                     \
        } else if (DIVISOR == 12) {                   \
            using blob_t = int3;                      \
            return __VA_ARGS__();                     \
        } else if (DIVISOR == 8) {                    \
            using blob_t = int2;                      \
            return __VA_ARGS__();                     \
        } else if (DIVISOR == 4) {                    \
            using blob_t = int;                       \
            return __VA_ARGS__();                     \
        } else if (DIVISOR == 2) {                    \
            using blob_t = int16_t;                   \
            return __VA_ARGS__();                     \
        } else {                                      \
            using blob_t = uint8_t;                   \
            return __VA_ARGS__();                     \
        }                                             \
    }()
#endif

template <typename T, int N>
struct MiniVecHash {
public:
    FN_SPECIFIERS
    uint64_t operator()(const MiniVec<T, N>& key) const {
        uint64_t hash = UINT64_C(14695981039346656037);
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int i = 0; i < N; ++i) {
            hash ^= static_cast<uint64_t>(key[i]);
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

template <typename T, int N>
struct MiniVecEq {
public:
    FN_SPECIFIERS
    bool operator()(const MiniVec<T, N>& lhs, const MiniVec<T, N>& rhs) const {
        bool is_equal = true;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int i = 0; i < N; ++i) {
            is_equal = is_equal && (lhs[i] == rhs[i]);
        }
        return is_equal;
    }
};
