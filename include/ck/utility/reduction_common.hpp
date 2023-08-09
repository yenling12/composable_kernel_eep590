// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/reduction_enums.hpp"

namespace ck {

struct float_equal_one
{
    template <class T>
    __host__ __device__ inline bool operator()(T x)
    {
        return x <= static_cast<T>(1.0f) and x >= static_cast<T>(1.0f);
    };
};

struct float_equal_zero
{
    template <class T>
    __host__ __device__ inline bool operator()(T x)
    {
        return x <= static_cast<T>(0.0f) and x >= static_cast<T>(0.0f);
    };
};

template <index_t N>
static constexpr __device__ index_t get_shift()
{
    return (get_shift<N / 2>() + 1);
};

template <>
constexpr __device__ index_t get_shift<1>()
{
    return (0);
}

template<typename T>
__host__ __device__ void waveReduceSum(T& src)
{
        T val;
	index_t sumVal = 0;
       //	= __builtin_amdgcn_readlane(src,63);
        asm volatile("\n \
            v_add_f32 %0, %1, %1 row_shr:1 bound_ctrl:0\n \
            v_add_f32 %0, %1, %0 row_shr:2 bound_ctrl:0\n \
            v_add_f32 %0, %1, %0 row_shr:3 bound_ctrl:0\n \
            v_nop\n \
            v_nop\n \
            v_add_f32 %0, %0, %0 row_shr:4 bound_ctrl:0\n \
            v_nop\n \
            v_nop\n \
            v_add_f32 %0, %0, %0 row_shr:8 bound_ctrl:0\n \
            v_nop\n \
            v_nop\n \
            v_add_f32 %1, %0, %0 row_bcast:15 row_mask:0xa\n \
	    v_nop\n \
	    v_nop\n \
            v_add_f32 %1, %1, %1 row_bcast:31 row_mask:0xc\n \
	    v_nop\n \
	    v_nop\n \
	    v_readlane_b32 %2, %1, 63\n \
            v_nop\n \
            v_nop\n \
            v_mov_b32 %1, %2\n \
            "
                    : "=v"(val)
                    : "v"(src), "s"(sumVal),
                      "0"(val));
}

} // namespace ck
