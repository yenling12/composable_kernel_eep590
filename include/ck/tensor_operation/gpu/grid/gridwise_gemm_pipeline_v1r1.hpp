// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"

namespace ck {

template <index_t NumPrefetch>
struct GridwiseGemmPipeline_v1r1;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_v1r1<1>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename FloatAB,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
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
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop,
                               void* p_shared,
                               bool bIsLoadAblock)
    {
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared) + 0, a_block_desc.GetElementSpaceSize());
        // preload data into LDS
        if(bIsLoadAblock)
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        if(bIsLoadAblock)
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        if(bIsLoadAblock)
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                if(bIsLoadAblock)
                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();
                if(bIsLoadAblock)
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<FloatAB*>(p_shared) + a_block_desc.GetElementSpaceSize() * (i + 1),
                    a_block_desc.GetElementSpaceSize());
                if(bIsLoadAblock)
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

} // namespace ck
