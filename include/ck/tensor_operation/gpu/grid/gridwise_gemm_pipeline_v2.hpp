// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {

template <bool AEnableLds = true, bool BEnableLds = true>
struct GridwiseGemmPipeline_v2;

template <>
struct GridwiseGemmPipeline_v2<false, true>
{
    static constexpr auto I0 = Number<0>{};

    __host__ __device__ static constexpr bool IsSupported(const index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop % 2 == 0;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(const index_t num_loop)
    {
        return (num_loop / 2) > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0);
        auto a_block_buf_switch           = a_block_buf;

        // global read 0
        a_blockwise_copy.Run(
            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // move to 1
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // global Read 1
        a_blockwise_copy.Run(
            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);

        // LDS write 0
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
        // global Read 1
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
#if CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT
                __builtin_amdgcn_iglp_opt(CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT);
#endif

                block_sync_lds();

                // GEMM i
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                // move to i + 2
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // global read i + 2
                a_block_buf = a_block_buf_switch;
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);

                // LDS write i + 1
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                // global read i + 2
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                ++i;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();

            // LDS write num_loop - 1
            a_block_buf = a_block_buf_switch;
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            block_sync_lds();

            // GEMM num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

template <>
struct GridwiseGemmPipeline_v2<true, false>
{
    static constexpr auto I0 = Number<0>{};

    __host__ __device__ static constexpr bool IsSupported(const index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop % 2 == 0;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(const index_t num_loop)
    {
        return (num_loop / 2) > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0);
        auto b_block_buf_switch           = b_block_buf;

        // global read 0
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);

        // move to 1
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // LDS write 0
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        // global Read 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        
        // global Read 1
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);
        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
#if CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT
                __builtin_amdgcn_iglp_opt(CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT);
#endif

                block_sync_lds();

                // GEMM i
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                // move to i + 2
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // LDS write i + 1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                // global read i + 2
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                // global read i + 2
                b_block_buf = b_block_buf_switch;
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);

                ++i;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();

            // LDS write num_loop - 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_block_buf = b_block_buf_switch;

            block_sync_lds();

            // GEMM num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

template <>
struct GridwiseGemmPipeline_v2<false, false>
{
    static constexpr auto I0 = Number<0>{};

    __host__ __device__ static constexpr bool IsSupported(const index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop % 2 == 0;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(const index_t num_loop)
    {
        return (num_loop / 2) > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0);
        auto a_block_buf_switch           = a_block_buf;
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0);
        auto b_block_buf_switch           = b_block_buf;

        // global read 0
        a_blockwise_copy.Run(
            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);
        // move to 1
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // global Read 1
        a_blockwise_copy.Run(
            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
#if CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT
                __builtin_amdgcn_iglp_opt(CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT);
#endif
                // GEMM i
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // move to i + 2
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // global read i + 2
                a_block_buf = a_block_buf_switch;
                b_block_buf = b_block_buf_switch;
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);
                ++i;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            // GEMM num_loop - 2
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            // LDS write num_loop - 1
            a_block_buf = a_block_buf_switch;
            b_block_buf = b_block_buf_switch;

            // GEMM num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

template <>
struct GridwiseGemmPipeline_v2<true, true>
{
    static constexpr auto I0 = Number<0>{};
    
    __host__ __device__ static constexpr bool IsSupported(const index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop % 2 == 0;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(const index_t num_loop)
    {
        return (num_loop / 2) > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // global read 0
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // move to 1
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // LDS write 0
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        // global Read 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

        // LDS write 0
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
        // global Read 1
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
#if CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT
                __builtin_amdgcn_iglp_opt(CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT);
#endif

                block_sync_lds();

                // GEMM i
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                // move to i + 2
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // LDS write i + 1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                // global read i + 2
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                // LDS write i + 1
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                // global read i + 2
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                ++i;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();

            // LDS write num_loop - 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            block_sync_lds();

            // GEMM num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

} // namespace ck
