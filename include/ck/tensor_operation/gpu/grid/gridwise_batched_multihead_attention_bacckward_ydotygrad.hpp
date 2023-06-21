// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_softmax.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_dropout.hpp"

namespace ck {

template <typename InputDataType,
          typename GemmDataType,
          typename FloatGemmAcc,
          typename FloatORS,
          typename CGridDesc_M_N,
          typename ORSGridDesc_M,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t B1K1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          index_t ABlockLdsExtraM,
          index_t BBlockLdsExtraN,
          bool Deterministic>
struct GridwiseBatchedMultiheadAttentionBackward_YDotYGrad
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};
    static constexpr auto I8 = Number<8>{};
    static constexpr auto I9 = Number<9>{};

    static constexpr auto WaveSize = 64;
    // K1 should be Number<...>
    // Gemm0
    static constexpr auto AK0 = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0 = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    // Gemm1
    static constexpr auto B1K0 = Number<Gemm1KPerBlock / B1K1Value>{};
    static constexpr auto B1K1 = Number<B1K1Value>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n,
                                                            const Block2CTileMap& block_2_ctile_map)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        if(!block_2_ctile_map.CheckValidity(c_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / Gemm1NPerBlock;

        const auto y_grid_desc_mblock_mperblock_oblock_operblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return y_grid_desc_mblock_mperblock_oblock_operblock;
    }

    __host__ __device__ static constexpr auto
    MakeORSGridDescriptor_MBlock_MRepeat_NWave_MPerXdl(const ORSGridDesc_M& lse_grid_desc_m)
    {
        const index_t M      = lse_grid_desc_m.GetLength(I0);
        const index_t MBlock = M / MPerBlock;

        const auto lse_grid_desc_mblock_mrepeat_mwave_mperxdl = transform_tensor_descriptor(
            lse_grid_desc_m,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{}))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1>{}));

        return lse_grid_desc_mblock_mrepeat_mwave_mperxdl;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, Gemm1NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }

    using YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock = remove_cvref_t<decltype(
        MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}))>;

    // S / dP Gemm (type 1 rcr)
    struct Gemm0
    {
        // A matrix in LDS memory, dst of blockwise copy
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        template <typename ABlockDesc_AK0_M_AK1>
        __host__ __device__ static constexpr auto
        MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
        {
            constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, MWaves, MPerXdl>(
                ABlockDesc_AK0_M_AK1{});
        }

        template <typename BBlockDesc_BK0_N_BK1>
        __host__ __device__ static constexpr auto
        MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
        {
            constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, NWaves, NPerXdl>(
                BBlockDesc_BK0_N_BK1{});
        }

        static constexpr index_t KPack =
            math::max(math::lcm(AK1, BK1),
                      MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        // Blockwise gemm with transposed XDL output
        using BlockwiseGemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            decltype(MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_ak0_m_ak1)),
            decltype(MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(b_block_desc_bk0_n_bk1)),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            true>; // TransposeC

        static constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        static constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);
    };

    template <index_t BlockSize_, index_t BlockSliceLength_M_, index_t BlockSliceLength_O_>
    struct YDotYGrad_M_O_
    {
        static_assert(BlockSize_ == BlockSliceLength_M_);
        static constexpr auto ThreadSliceLength_M   = Number<1>{};
        static constexpr index_t SrcScalarPerVector = 16 / sizeof(InputDataType);
        static constexpr auto ThreadClusterLength_O = Number<1>{};
        static constexpr auto ThreadClusterLength_M = Number<BlockSize_ / ThreadClusterLength_O>{};
        static constexpr auto ThreadSliceLength_O   = Number<BlockSliceLength_O_>{};

        static_assert(ThreadClusterLength_O * ThreadSliceLength_O == BlockSliceLength_O_, "");
        static_assert(ThreadClusterLength_M * ThreadSliceLength_M == BlockSliceLength_M_, "");

        using SrcBufType = StaticBuffer<AddressSpaceEnum::Vgpr,
                                        FloatGemmAcc,
                                        ThreadSliceLength_M * ThreadSliceLength_O,
                                        true>;

        using DstBufType =
            StaticBuffer<AddressSpaceEnum::Vgpr, FloatGemmAcc, ThreadSliceLength_M, true>;
    };
    using YDotYGrad_M_O = YDotYGrad_M_O_<BlockSize, MPerBlock, Gemm1NPerBlock>;

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        return MPerBlock * sizeof(FloatGemmAcc);
    }

    __device__ static void test() {}
    template <typename Block2CTileMap>
    __device__ static void Run(const InputDataType* __restrict__ p_y_grid,
                               const InputDataType* __restrict__ p_ygrad_grid,
                               FloatORS* __restrict__ p_ors_grid,
                               void* __restrict__ p_shared,
                               const YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock&
                                   y_grid_desc_mblock_mperblock_oblock_operblock,
                               const ORSGridDesc_M& ors_grid_desc_m,
                               const Block2CTileMap& block_2_ctile_map,
                               const index_t block_idx_m)
    {
        const auto y_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_grid, y_grid_desc_mblock_mperblock_oblock_operblock.GetElementSpaceSize());
        const auto ygrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_ygrad_grid, y_grid_desc_mblock_mperblock_oblock_operblock.GetElementSpaceSize());

        auto ors_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_ors_grid, ors_grid_desc_m.GetElementSpaceSize());

        // divide block work by [M, O]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(y_grid_desc_mblock_mperblock_oblock_operblock.GetLength(I0),
                          y_grid_desc_mblock_mperblock_oblock_operblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_work_idx_m = Deterministic ? block_idx_m : block_work_idx[I0];

        constexpr auto ors_thread_desc_mblock_mrepeat_mwave_mperxdl =
            make_naive_tensor_descriptor_packed(make_tuple(I1, I1));

        constexpr auto y_thread_desc_m0_m1_o0_o1 = make_naive_tensor_descriptor_packed(make_tuple(
            I1, YDotYGrad_M_O::ThreadSliceLength_M, I1, YDotYGrad_M_O::ThreadSliceLength_O));

        constexpr auto y_thread_cluster_desc =
            make_cluster_descriptor(Sequence<I1,
                                             YDotYGrad_M_O::ThreadClusterLength_M,
                                             I1,
                                             YDotYGrad_M_O::ThreadClusterLength_O>{},
                                    Sequence<0, 1, 2, 3>{});
        const auto y_thread_cluster_idx =
            y_thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto y_thread_data_on_block_idx =
            y_thread_cluster_idx * y_thread_desc_m0_m1_o0_o1.GetLengths();

        // if(get_thread_global_1d_id() == 1)
        // {
        //     printf("y_thread_data_on_block_idx:{ %d, %d, %d,%d}, get_thread_local_1d_id: %d\n",
        //            y_thread_data_on_block_idx[I0],
        //            y_thread_data_on_block_idx[I1],
        //            y_thread_data_on_block_idx[I2],
        //            y_thread_data_on_block_idx[I3],
        //            get_thread_local_1d_id());
        // }
        const auto y_thread_data_on_grid_idx =
            make_multi_index(
                block_work_idx_m, I0, I0 /* all WGs start from o_block_idx = 0 */, I0) +
            y_thread_data_on_block_idx;

        // performs double duty for both y and ygrad
        auto yygrad_threadwise_copy = ThreadwiseTensorSliceTransfer_v2<
            InputDataType,
            FloatGemmAcc,
            YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock,
            decltype(y_thread_desc_m0_m1_o0_o1),
            decltype(y_thread_desc_m0_m1_o0_o1.GetLengths()),
            Sequence<0, 1, 2, 3>,
            3,                                 // SrcVectorDim
            YDotYGrad_M_O::SrcScalarPerVector, // SrcScalarPerVector
            1,                                 // SrcScalarStrideInVector
            true /* ResetCoordAfterRun */,
            false /* InvalidElementAsNaN */>(y_grid_desc_mblock_mperblock_oblock_operblock,
                                             y_thread_data_on_grid_idx);

        auto y_thread_buf                 = typename YDotYGrad_M_O::SrcBufType{};
        auto ygrad_thread_buf             = typename YDotYGrad_M_O::SrcBufType{};
        auto y_dot_ygrad_thread_accum_buf = typename YDotYGrad_M_O::DstBufType{};
        auto y_dot_ygrad_block_accum_buf  = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatGemmAcc*>(p_shared), MPerBlock);

        if constexpr(Deterministic)
        {
            block_sync_lds();
        }

        // clear accum buffers
        y_dot_ygrad_thread_accum_buf.Clear();
        y_dot_ygrad_block_accum_buf.Clear();

        index_t oblock_idx = 0;
        do
        {
            yygrad_threadwise_copy.Run(y_grid_desc_mblock_mperblock_oblock_operblock,
                                       y_grid_buf,
                                       y_thread_desc_m0_m1_o0_o1,
                                       make_tuple(I0, I0, I0, I0),
                                       y_thread_buf);
            yygrad_threadwise_copy.Run(y_grid_desc_mblock_mperblock_oblock_operblock,
                                       ygrad_grid_buf,
                                       y_thread_desc_m0_m1_o0_o1,
                                       make_tuple(I0, I0, I0, I0),
                                       ygrad_thread_buf);

            static_for<0, YDotYGrad_M_O::ThreadSliceLength_M, 1>{}([&](auto iM) {
                static_for<0, YDotYGrad_M_O::ThreadSliceLength_O, 1>{}([&](auto iO) {
                    constexpr auto offset =
                        y_thread_desc_m0_m1_o0_o1.CalculateOffset(make_multi_index(I0, iM, I0, iO));
                    y_dot_ygrad_thread_accum_buf(iM) +=
                        y_thread_buf[Number<offset>{}] * ygrad_thread_buf[Number<offset>{}];
                });
            });

            yygrad_threadwise_copy.MoveSrcSliceWindow(y_grid_desc_mblock_mperblock_oblock_operblock,
                                                      make_multi_index(0, 0, 1, 0));

            oblock_idx++;
        } while(oblock_idx < y_grid_desc_mblock_mperblock_oblock_operblock.GetLength(I2));

        auto ors_grid_desc_mblock_mrepeat_mwave_mperxdl =
            MakeORSGridDescriptor_MBlock_MRepeat_NWave_MPerXdl(ors_grid_desc_m);

        auto ors_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            FloatORS,
            decltype(ors_thread_desc_mblock_mrepeat_mwave_mperxdl),
            decltype(ors_grid_desc_mblock_mrepeat_mwave_mperxdl),
            ck::tensor_operation::element_wise::PassThrough,
            Sequence<1, 1>,
            Sequence<0, 1>,
            1,
            1,
            InMemoryDataOperationEnum::Set,
            1,
            false>{ors_grid_desc_mblock_mrepeat_mwave_mperxdl,
                   make_multi_index(block_work_idx_m,          // mblock
                                    get_thread_local_1d_id()), // mperxdl
                   ck::tensor_operation::element_wise::PassThrough{}};

        // copy from VGPR to Global
        ors_thread_copy_vgpr_to_global.Run(ors_thread_desc_mblock_mrepeat_mwave_mperxdl,
                                           make_tuple(I0, I0),
                                           y_dot_ygrad_thread_accum_buf,
                                           ors_grid_desc_mblock_mrepeat_mwave_mperxdl,
                                           ors_grid_buf);

        ignore = ors_thread_copy_vgpr_to_global;
        ignore = ors_grid_desc_mblock_mrepeat_mwave_mperxdl;
    }
};

} // namespace ck
