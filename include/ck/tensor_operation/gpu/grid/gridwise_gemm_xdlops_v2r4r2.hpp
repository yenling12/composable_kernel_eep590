// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

namespace ck {

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename Block2CTileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_xdlops_v2r4r2_simplified(typename GridwiseGemm::Argument karg,
                                             const Block2CTileMap& b2c_map,
                                             const AElementwiseOperation a_element_op,
                                             const BElementwiseOperation b_element_op,
                                             const CElementwiseOperation c_element_op)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    constexpr index_t shared_size = GridwiseGemm::SharedMemTrait::lds_size;

    __shared__ uint8_t p_shared[shared_size];

    GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation>(
        karg, static_cast<void*>(p_shared), b2c_map, a_element_op, b_element_op, c_element_op);
#else
    ignore = karg;
    ignore = b2c_map;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename FloatC,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t K1Value,
          index_t MRepeat_,
          index_t NRepeat_,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool AEnableLds,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          bool BEnableLds,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          PipelineVersion PipelineVer = PipelineVersion::v1,
          typename ComputeType        = FloatC>
struct GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2
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

    // K1 should be Number<...>
    static constexpr auto K1        = Number<K1Value>{};
    static constexpr auto AK1       = Number<K1Value>{};
    static constexpr auto BK1       = Number<K1Value>{};
    static constexpr auto KPerBlock = K1 * K0PerBlock;
    static constexpr auto M01       = 1;
    static constexpr auto N01       = 1;

    static constexpr auto gemm_padder =
        tensor_operation::device::GemmPadder<GemmSpec, index_t, index_t, index_t>{
            MPerBlock, NPerBlock, K1* K0PerBlock};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe =
        remove_cvref_t<decltype(GridwiseGemmPipeline_Selector<PipelineVer,
                                                              NumGemmKPrefetchStage,
                                                              AEnableLds,
                                                              BEnableLds,
                                                              LoopSched>())>;

    static constexpr index_t KPack = math::max(
        math::lcm(AK1, BK1), MfmaSelector<FloatA, MPerXDL, NPerXDL>::selected_mfma.k_per_blk);
    static constexpr index_t MPack    = AEnableLds ? 1
                                        : is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value
                                            ? ABlockTransferSrcScalarPerVector
                                            : 1;
    static constexpr index_t NPack    = BEnableLds ? 1
                                        : is_same<tensor_layout::gemm::RowMajor, BLayout>::value
                                            ? BBlockTransferSrcScalarPerVector
                                            : 1;
    static constexpr auto xdlops_gemm = XdlopsGemm<FloatA, MPerXDL, NPerXDL, KPack, MPack, NPack>{};
    static constexpr auto K0PerXDL    = KPack / K1;
    static constexpr auto XDL_KIter   = KPack / xdlops_gemm.K1PerXdlops;
    static constexpr auto KPerXDL     = XDL_KIter * xdlops_gemm.KPerXdlops;
    static constexpr auto KThread     = xdlops_gemm.KThreadPerXdlops;
    static constexpr auto KRepeat     = KPerBlock / KPerXDL;
    static constexpr auto MRepeat     = MRepeat_ / MPack;
    static constexpr auto NRepeat     = NRepeat_ / NPack;

    static constexpr auto MWaves = MPerBlock / MRepeat / MPerXDL / MPack;
    static constexpr auto NWaves = NPerBlock / NRepeat / NPerXDL / NPack;

    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        const FloatA* p_a_grid;
        const FloatB* p_b_grid;
        FloatC* p_c_grid;
        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t MPadded;
        index_t NPadded;
        index_t KPadded;
        index_t K0Padded;
        index_t k_batch;

        Argument(const FloatA* p_a_grid_,
                 const FloatB* p_b_grid_,
                 FloatC* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_,
                 index_t MPadded_,
                 index_t NPadded_,
                 index_t KPadded_,
                 index_t K0Padded_,
                 index_t k_batch_)
            : p_a_grid(p_a_grid_),
              p_b_grid(p_b_grid_),
              p_c_grid(p_c_grid_),
              M(M_),
              N(N_),
              K(K_),
              StrideA(StrideA_),
              StrideB(StrideB_),
              StrideC(StrideC_),
              MPadded(MPadded_),
              NPadded(NPadded_),
              KPadded(KPadded_),
              K0Padded(K0Padded_),
              k_batch(k_batch_)
        {
        }

        void Print() const
        {
            std::cout << "arg {"
                      << "M:" << M << ", "
                      << "N:" << N << ", "
                      << "K:" << K << ", "
                      << "SA:" << StrideA << ", "
                      << "SB:" << StrideB << ", "
                      << "SC:" << StrideC << ", "
                      << "MP:" << MPadded << ", "
                      << "NP:" << NPadded << ", "
                      << "KP:" << KPadded << ", "
                      << "K0Padded:" << K0Padded << ", "
                      << "KB:" << k_batch << "}" << std::endl;
        }
    };

    __host__ __device__ static auto CalculateGridSize(const Argument& karg)
    {
        return std::make_tuple(math::integer_divide_ceil(karg.N, NPerBlock),
                               math::integer_divide_ceil(karg.M, MPerBlock),
                               karg.k_batch);
    }

    // prefer this to be called on host
    __host__ __device__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static auto CalculateK0Padded(index_t K, index_t K_Batch = 1)
    {
        // k_batch * k0 * k0_per_block * k1
        auto K_t = K_Batch * K0PerBlock * K1;
        return (K + K_t - 1) / K_t * K0PerBlock;
    }

    __host__ __device__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        auto K0Padded = CalculateK0Padded(K, K_Batch);
        return K_Batch * K0Padded * K1;
    }

    __host__ __device__ static auto MakeAGridDescriptor(index_t M,
                                                        index_t MPad,
                                                        index_t K,
                                                        index_t StrideA,
                                                        index_t KBlock,
                                                        index_t K0Padded,
                                                        index_t KPad)
    {
        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {

            const auto a_grid_desc_mpad_kpad =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_right_pad_transform(M, MPad - M),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto MBlock = MPad / MPerBlock;

            // const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            if constexpr(AEnableLds)
            {
                // KBlock -> K0Padded -> MBlock -> MPerblock -> K1
                return transform_tensor_descriptor(
                    a_grid_desc_mpad_kpad,
                    make_tuple(make_unmerge_transform(make_tuple(KBlock, K0Padded, K1)),
                               make_unmerge_transform(make_tuple(MBlock, MPerBlock))),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0, 1, 4>{}, Sequence<2, 3>{}));
            }
            else
            {
                // KTile * KRepeat
                const auto KXDL = K0Padded * K1 / KPerXDL;

                if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
                {
                    // 0   1     0        1         2        3         4        5         6      7 8
                    // M - K <-> KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - MPerXDL -
                    // K0PerXDL - K1
                    return transform_tensor_descriptor(
                        a_grid_desc_mpad_kpad,
                        make_tuple(make_unmerge_transform(
                                       make_tuple(I1, KXDL, Number<KThread>{}, K0PerXDL, K1)),
                                   make_unmerge_transform(make_tuple(
                                       MBlock, MRepeat, Number<MWaves>{}, Number<MPerXDL>{}))),
                        make_tuple(Sequence<1>{}, Sequence<0>{}),
                        make_tuple(Sequence<0, 1, 5, 7, 8>{}, Sequence<2, 3, 4, 6>{}));
                }
                else
                {
                    // 0   1     0        1      2        3         4        5         6       7 8
                    // M - K <-> KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - KPack -
                    // MPerXDL - MPack
                    return transform_tensor_descriptor(
                        a_grid_desc_mpad_kpad,
                        make_tuple(make_unmerge_transform(
                                       make_tuple(I1, KXDL, Number<KThread>{}, Number<KPack>{})),
                                   make_unmerge_transform(make_tuple(MBlock,
                                                                     Number<MRepeat>{},
                                                                     Number<MWaves>{},
                                                                     Number<MPerXDL>{},
                                                                     Number<MPack>{}))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0, 1, 5, 6>{}, Sequence<2, 3, 4, 7, 8>{}));
                }
            }
        }
        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                          GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {
            const auto a_grid_desc_mpad_k = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_right_pad_transform(M, MPad - M), make_pass_through_transform(K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto MBlock = MPad / MPerBlock;
            // const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            if constexpr(AEnableLds)
            {
                return transform_tensor_descriptor(
                    a_grid_desc_mpad_k,
                    make_tuple(make_unmerge_transform(make_tuple(KBlock, K0Padded, K1)),
                               make_unmerge_transform(make_tuple(MBlock, MPerBlock))),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0, 1, 4>{}, Sequence<2, 3>{}));
            }
            else
            {
                const auto KXDL = K0Padded * K1 / KPerXDL;
                if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
                {
                    // 0   1     0        1         2        3         4        5         6      7 8
                    // M - K <-> KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - MPerXDL -
                    // K0PerXDL - K1
                    return transform_tensor_descriptor(
                        a_grid_desc_mpad_k,
                        make_tuple(make_unmerge_transform(
                                       make_tuple(I1, KXDL, Number<KThread>{}, K0PerXDL, K1)),
                                   make_unmerge_transform(make_tuple(
                                       MBlock, MRepeat, Number<MWaves>{}, Number<MPerXDL>{}))),
                        make_tuple(Sequence<1>{}, Sequence<0>{}),
                        make_tuple(Sequence<0, 1, 5, 7, 8>{}, Sequence<2, 3, 4, 6>{}));
                }
                else
                {
                    // 0   1     0        1      2        3         4        5         6       7 8
                    // M - K <-> KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - KPack -
                    // MPerXDL - MPack
                    return transform_tensor_descriptor(
                        a_grid_desc_mpad_k,
                        make_tuple(make_unmerge_transform(
                                       make_tuple(I1, KXDL, Number<KThread>{}, Number<KPack>{})),
                                   make_unmerge_transform(make_tuple(MBlock,
                                                                     Number<MRepeat>{},
                                                                     Number<MWaves>{},
                                                                     Number<MPerXDL>{},
                                                                     Number<MPack>{}))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0, 1, 5, 6>{}, Sequence<2, 3, 4, 7, 8>{}));
                }
            }
        }
        else
        {
            const auto MBlock = M / MPerBlock;
            if constexpr(AEnableLds)
            {
                return transform_tensor_descriptor(
                    a_grid_desc_m_k,
                    make_tuple(make_unmerge_transform(make_tuple(KBlock, K0Padded, K1)),
                               make_unmerge_transform(make_tuple(MBlock, MPerBlock))),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0, 1, 4>{}, Sequence<2, 3>{}));
            }
            else
            {
                const auto KXDL = K0Padded * K1 / KPerXDL;
                if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
                {
                    // 0   1     0        1         2        3         4        5         6      7 8
                    // M - K <-> KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - MPerXDL -
                    // K0PerXDL - K1
                    return transform_tensor_descriptor(
                        a_grid_desc_m_k,
                        make_tuple(make_unmerge_transform(
                                       make_tuple(I1, KXDL, Number<KThread>{}, K0PerXDL, K1)),
                                   make_unmerge_transform(make_tuple(
                                       MBlock, MRepeat, Number<MWaves>{}, Number<MPerXDL>{}))),
                        make_tuple(Sequence<1>{}, Sequence<0>{}),
                        make_tuple(Sequence<0, 1, 5, 7, 8>{}, Sequence<2, 3, 4, 6>{}));
                }
                else
                {
                    // 0   1     0        1      2        3         4        5         6       7 8
                    // M - K <-> KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - KPack -
                    // MPerXDL - MPack
                    return transform_tensor_descriptor(
                        a_grid_desc_m_k,
                        make_tuple(make_unmerge_transform(
                                       make_tuple(I1, KXDL, Number<KThread>{}, Number<KPack>{})),
                                   make_unmerge_transform(make_tuple(MBlock,
                                                                     Number<MRepeat>{},
                                                                     Number<MWaves>{},
                                                                     Number<MPerXDL>{},
                                                                     Number<MPack>{}))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0, 1, 5, 6>{}, Sequence<2, 3, 4, 7, 8>{}));
                }
            }
        }
    }

    __host__ __device__ static auto MakeBGridDescriptor(index_t K,
                                                        index_t NPad,
                                                        index_t N,
                                                        index_t StrideB,
                                                        index_t KBlock,
                                                        index_t K0Padded,
                                                        index_t KPad)
    {
        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {

            const auto NBlock = NPad / NPerBlock;

            const auto b_grid_desc_kpad_npad =
                transform_tensor_descriptor(b_grid_desc_k_n,
                                            make_tuple(make_right_pad_transform(K, KPad - K),
                                                       make_right_pad_transform(N, NPad - N)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            // const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;
            if constexpr(BEnableLds)
            {
                return transform_tensor_descriptor(
                    b_grid_desc_kpad_npad,
                    make_tuple(make_unmerge_transform(make_tuple(KBlock, K0Padded, K1)),
                               make_unmerge_transform(make_tuple(NBlock, NPerBlock))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 1, 4>{}, Sequence<2, 3>{}));
            }
            else
            {
                // K-Major
                const auto KXDL = K0Padded * K1 / KPerXDL;

                if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
                {
                    // 0   1     0        1         2        3         4        5         6      7 8
                    // K - N <-> KBlock - KXDL - NBlock - NRepeat - NWaves - KThread - NPerXDL -
                    // K0PerXDL - K1
                    return transform_tensor_descriptor(
                        b_grid_desc_kpad_npad,
                        make_tuple(
                            make_unmerge_transform(
                                make_tuple(I1, KXDL, Number<KThread>{}, Number<K0PerXDL>{}, K1)),
                            make_unmerge_transform(make_tuple(
                                NBlock, Number<NRepeat>{}, Number<NWaves>{}, Number<NPerXDL>{}))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0, 1, 5, 7, 8>{}, Sequence<2, 3, 4, 6>{}));
                }
                else
                {
                    // 0   1     0        1      2        3         4        5         6       7 8
                    // K - N <-> KBlock - KXDL - NBlock - NRepeat - NWaves - KThread - KPack -
                    // NPerXDL - NPack
                    return transform_tensor_descriptor(
                        b_grid_desc_kpad_npad,
                        make_tuple(make_unmerge_transform(
                                       make_tuple(I1, KXDL, Number<KThread>{}, Number<KPack>{})),
                                   make_unmerge_transform(make_tuple(NBlock,
                                                                     Number<NRepeat>{},
                                                                     Number<NWaves>{},
                                                                     Number<NPerXDL>{},
                                                                     Number<NPack>{}))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0, 1, 5, 6>{}, Sequence<2, 3, 4, 7, 8>{}));
                }
            }
        }
        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                          GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {
            const auto b_grid_desc_k_npad = transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_pass_through_transform(K), make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto NBlock = NPad / NPerBlock;

            if constexpr(BEnableLds)
            {
                return transform_tensor_descriptor(
                    b_grid_desc_k_npad,
                    make_tuple(make_unmerge_transform(make_tuple(KBlock, K0Padded, K1)),
                               make_unmerge_transform(make_tuple(NBlock, NPerBlock))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 1, 4>{}, Sequence<2, 3>{}));
            }
            else
            {
                const auto KXDL = K0Padded * K1 / KPerXDL;

                if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
                {
                    // 0   1     0        1         2        3         4        5         6      7 8
                    // K - N <-> KBlock - KXDL - NBlock - NRepeat - NWaves - KThread - NPerXDL -
                    // K0PerXDL - K1
                    return transform_tensor_descriptor(
                        b_grid_desc_k_npad,
                        make_tuple(
                            make_unmerge_transform(
                                make_tuple(I1, KXDL, Number<KThread>{}, Number<K0PerXDL>{}, K1)),
                            make_unmerge_transform(make_tuple(
                                NBlock, Number<NRepeat>{}, Number<NWaves>{}, Number<NPerXDL>{}))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0, 1, 5, 7, 8>{}, Sequence<2, 3, 4, 6>{}));
                }
                else
                {
                    // 0   1     0        1      2        3         4        5         6       7 8
                    // K - N <-> KBlock - KXDL - NBlock - NRepeat - NWaves - KThread - KPack -
                    // NPerXDL - NPack
                    return transform_tensor_descriptor(
                        b_grid_desc_k_npad,
                        make_tuple(make_unmerge_transform(
                                       make_tuple(I1, KXDL, Number<KThread>{}, Number<KPack>{})),
                                   make_unmerge_transform(make_tuple(NBlock,
                                                                     Number<NRepeat>{},
                                                                     Number<NWaves>{},
                                                                     Number<NPerXDL>{},
                                                                     Number<NPack>{}))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0, 1, 5, 6>{}, Sequence<2, 3, 4, 7, 8>{}));
                }
            }
        }
        else
        {
            const auto NBlock = N / NPerBlock;

            if constexpr(BEnableLds)
            {
                return transform_tensor_descriptor(
                    b_grid_desc_k_n,
                    make_tuple(make_unmerge_transform(make_tuple(KBlock, K0Padded, K1)),
                               make_unmerge_transform(make_tuple(NBlock, NPerBlock))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 1, 4>{}, Sequence<2, 3>{}));
            }
            else
            {
                const auto KXDL = K0Padded * K1 / KPerXDL;

                if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
                {
                    // 0   1     0        1         2        3         4        5         6      7 8
                    // K - N <-> KBlock - KXDL - NBlock - NRepeat - NWaves - KThread - NPerXDL -
                    // K0PerXDL - K1
                    return transform_tensor_descriptor(
                        b_grid_desc_k_n,
                        make_tuple(
                            make_unmerge_transform(
                                make_tuple(I1, KXDL, Number<KThread>{}, Number<K0PerXDL>{}, K1)),
                            make_unmerge_transform(make_tuple(
                                NBlock, Number<NRepeat>{}, Number<NWaves>{}, Number<NPerXDL>{}))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0, 1, 5, 7, 8>{}, Sequence<2, 3, 4, 6>{}));
                }
                else
                {
                    // 0   1     0        1      2        3         4        5         6       7 8
                    // K - N <-> KBlock - KXDL - NBlock - NRepeat - NWaves - KThread - KPack -
                    // NPerXDL - NPack
                    return transform_tensor_descriptor(
                        b_grid_desc_k_n,
                        make_tuple(make_unmerge_transform(
                                       make_tuple(I1, KXDL, Number<KThread>{}, Number<KPack>{})),
                                   make_unmerge_transform(make_tuple(NBlock,
                                                                     Number<NRepeat>{},
                                                                     Number<NWaves>{},
                                                                     Number<NPerXDL>{},
                                                                     Number<NPack>{}))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0, 1, 5, 6>{}, Sequence<2, 3, 4, 7, 8>{}));
                }
            }
        }
    }

    __host__ __device__ static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        return gemm_padder.PadCDescriptor_M_N(c_grid_desc_m_n);
    }

    // Describe how data store to (LDS/VGPR) buffer from Global memory
    __host__ __device__ static constexpr auto MakeABlockDescriptor()
    {
        constexpr auto a_block_desc = [&]() {
            if constexpr(AEnableLds)
            {
                constexpr auto max_lds_align = K1;

                if constexpr(ABlockLdsExtraM)
                {
                    // Set KBlock & MBlock Dimension to 1
                    return make_naive_tensor_descriptor(
                        make_tuple(I1, Number<K0PerBlock>{}, I1, Number<MPerBlock>{}, K1),
                        make_tuple(Number<K0PerBlock>{} * Number<MPerBlock + 1>{} * K1,
                                   Number<MPerBlock + 1>{} * K1,
                                   Number<MPerBlock + 1>{} * K1,
                                   K1,
                                   I1));
                }
                else
                {
                    return make_naive_tensor_descriptor_aligned(
                        make_tuple(I1, Number<K0PerBlock>{}, I1, Number<MPerBlock>{}, K1),
                        max_lds_align);
                }
            }
            else
            {
                if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
                {
                    // 0       1        2       3        4       5        6        7         8
                    // KBlock->KRepeat->MBlock->MRepeat->MWaves->KThread->MPerXDL->K0PerXDL->K1
                    return make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                                          Number<KRepeat>{},
                                                                          I1,
                                                                          Number<MRepeat>{},
                                                                          I1,
                                                                          I1,
                                                                          I1,
                                                                          Number<K0PerXDL>{},
                                                                          Number<K1>{}));
                }
                else
                {
                    // 0        1      2        3         4        5         8       7         6
                    // KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - KPack - MPerXDL - MPack
                    return make_naive_tensor_descriptor(
                        make_tuple(I1,
                                   Number<KRepeat>{},
                                   I1,
                                   Number<MRepeat>{},
                                   I1,
                                   I1,
                                   Number<KPack>{},
                                   I1,
                                   Number<MPack>{}),
                        // Stride of
                        make_tuple(Number<KPack * MPack * MRepeat * KRepeat>{}, // KBlock
                                   Number<KPack * MPack * MRepeat>{},           // KXDL
                                   Number<KPack * MPack * MRepeat>{},           // MBlock
                                   Number<KPack * MPack>{},                     // MRepeat
                                   Number<KPack * MPack>{},                     // MWaves
                                   Number<KPack * MPack>{},                     // KThread
                                   I1,              // KPack [Fast change dimension]
                                   Number<KPack>{}, // MPerXDL
                                   Number<KPack>{}  // MPack
                                   ));
                }
            }
        }();

        return a_block_desc;
    }

    __host__ __device__ static constexpr auto MakeBBlockDescriptor()
    {
        constexpr auto b_block_desc = [&]() {
            if constexpr(BEnableLds)
            {
                constexpr auto max_lds_align = K1;

                if constexpr(BBlockLdsExtraN)
                {
                    // Set KBlock & MBlock Dimension to 1
                    return make_naive_tensor_descriptor(
                        make_tuple(I1, Number<K0PerBlock>{}, I1, Number<NPerBlock>{}, K1),
                        make_tuple(Number<K0PerBlock>{} * Number<NPerBlock + 1>{} * K1,
                                   Number<NPerBlock + 1>{} * K1,
                                   Number<NPerBlock + 1>{} * K1,
                                   K1,
                                   I1));
                }
                else
                {
                    return make_naive_tensor_descriptor_aligned(
                        make_tuple(I1, Number<K0PerBlock>{}, I1, Number<NPerBlock>{}, K1),
                        max_lds_align);
                }
            }
            else
            {
                if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
                {
                    // 0       1        2       3        4       5        6        7         8
                    // KBlock->KRepeat->NBlock->NRepeat->NWaves->KThread->NPerXDL->K0PerXDL->K1
                    return make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                                          Number<KRepeat>{},
                                                                          I1,
                                                                          Number<NRepeat>{},
                                                                          I1,
                                                                          I1,
                                                                          I1,
                                                                          Number<K0PerXDL>{},
                                                                          K1));
                }
                else
                {
                    // 0        1      2        3         4        5         8       7         6
                    // KBlock - KXDL - NBlock - NRepeat - NWaves - KThread - KPack - NPerXDL - NPack
                    return make_naive_tensor_descriptor(
                        make_tuple(I1,
                                   Number<KRepeat>{},
                                   I1,
                                   Number<NRepeat>{},
                                   I1,
                                   I1,
                                   Number<KPack>{},
                                   I1,
                                   Number<NPack>{}),
                        // Stride of
                        make_tuple(Number<KPack * NPack * NRepeat * KRepeat>{}, // KBlock
                                   Number<KPack * NPack * NRepeat>{},           // KXDL
                                   Number<KPack * NPack * NRepeat>{},           // NBlock
                                   Number<KPack * NPack>{},                     // NRepeat
                                   Number<KPack * NPack>{},                     // NWaves
                                   Number<KPack * NPack>{},                     // KThread
                                   I1,              // KPack [Fast change dimension]
                                   Number<KPack>{}, // NPerXDL
                                   Number<KPack>{}  // NVec
                                   ));
                }
            }
        }();

        return b_block_desc;
    }

    // Describe how data read from (LDS/VGPR) buffer
    template <typename ABlockDesc_>
    __host__ __device__ static constexpr auto MakeAWaveDescriptor(const ABlockDesc_&)
    {

        constexpr auto a_wave_desc = [&]() {
            if constexpr(AEnableLds)
            {
                // KBlock_AK0_MBlock_MPerBlock_AK1 ->
                // 0       1       2      3       4       5
                //                 I1     I1      I1
                // KRepeat-MRepeat-MWaves-KThread-MPerXDL-KPack

                constexpr auto a_wave_desc_temp = transform_tensor_descriptor(
                    ABlockDesc_{},
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_unmerge_transform(make_tuple(
                                   Number<KRepeat>{}, Number<KThread>{}, Number<K0PerXDL>{})),
                               make_unmerge_transform(make_tuple(Number<MRepeat>{},
                                                                 Number<MWaves>{},
                                                                 Number<MPerXDL>{},
                                                                 Number<MPack>{})),
                               make_pass_through_transform(Number<AK1>{})),
                    make_tuple(
                        Sequence<0>{}, Sequence<2>{}, Sequence<1>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<>{},
                               Sequence<>{},
                               Sequence<0, 3, 6>{},
                               Sequence<1, 2, 4, 5>{},
                               Sequence<7>{}));

                return transform_tensor_descriptor(
                    a_wave_desc_temp,
                    make_tuple(make_pass_through_transform(a_wave_desc_temp.GetLength(I0)),
                               make_pass_through_transform(a_wave_desc_temp.GetLength(I1)),
                               make_pass_through_transform(a_wave_desc_temp.GetLength(I2)),
                               make_pass_through_transform(a_wave_desc_temp.GetLength(I3)),
                               make_pass_through_transform(a_wave_desc_temp.GetLength(I4)),
                               make_pass_through_transform(a_wave_desc_temp.GetLength(I5)),
                               make_merge_transform(make_tuple(a_wave_desc_temp.GetLength(I6),
                                                               a_wave_desc_temp.GetLength(I7)))),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6, 7>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6>{}));
            }
            else
            {
                // KBlock->KRepeat->NBlock->MRepeat->MWaves->KThread->MPerXDL->K0PerXDL->K1
                // ->
                // KRepeat-MRepeat-MWaves-KThread-MPerXDL-KPack
                // This transform lead to non-constpexr error
#if 0
                return transform_tensor_descriptor(
                    ABlockDesc_{},
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_pass_through_transform(ABlockDesc_{}.GetLength(I1)),
                               make_pass_through_transform(ABlockDesc_{}.GetLength(I3)),
                               make_pass_through_transform(ABlockDesc_{}.GetLength(I4)),
                               make_pass_through_transform(ABlockDesc_{}.GetLength(I5)),
                               make_pass_through_transform(ABlockDesc_{}.GetLength(I6)),
                               make_merge_transform(make_tuple(ABlockDesc_{}.GetLength(I7),
                                                    ABlockDesc_{}.GetLength(I8)))),
                    make_tuple(Sequence<0>{},
                               Sequence<2>{},
                               Sequence<1>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6>{},
                               Sequence<7, 8>{}),
                    make_tuple(Sequence<>{},
                               Sequence<>{},
                               Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}));
#endif
#if 1
                if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
                {
                    // KBlock->KRepeat->MBlock->MRepeat->MWaves->KThread->MPerXDL->K0PerXDL->K1
                    // ->
                    // KRepeat-MRepeat-MWaves-KThread-MPerXDL-MPack-KPack
                    return make_naive_tensor_descriptor_packed(
                        make_tuple(ABlockDesc_{}.GetLength(I1),
                                   ABlockDesc_{}.GetLength(I3),
                                   ABlockDesc_{}.GetLength(I4),
                                   ABlockDesc_{}.GetLength(I5),
                                   ABlockDesc_{}.GetLength(I6),
                                   Number<MPack>{},
                                   ABlockDesc_{}.GetLength(I7) * ABlockDesc_{}.GetLength(I8)));
                }
                else
                {
                    // KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - KPack - MPerXDL - MPack
                    // ->
                    // KXDL - MRepeat - MWaves - KThread- MPerXDL- MPack - KPack
                    return transform_tensor_descriptor(
                        ABlockDesc_{},
                        make_tuple(make_freeze_transform(I0),
                                   make_freeze_transform(I0),
                                   make_pass_through_transform(ABlockDesc_{}.GetLength(I1)),
                                   make_pass_through_transform(ABlockDesc_{}.GetLength(I3)),
                                   make_pass_through_transform(ABlockDesc_{}.GetLength(I4)),
                                   make_pass_through_transform(ABlockDesc_{}.GetLength(I5)),
                                   make_pass_through_transform(ABlockDesc_{}.GetLength(I6)),
                                   make_pass_through_transform(ABlockDesc_{}.GetLength(I7)),
                                   make_pass_through_transform(ABlockDesc_{}.GetLength(I8))),
                        make_tuple(Sequence<0>{},
                                   Sequence<2>{},
                                   Sequence<1>{},
                                   Sequence<3>{},
                                   Sequence<4>{},
                                   Sequence<5>{},
                                   Sequence<6>{},
                                   Sequence<7>{},
                                   Sequence<8>{}),
                        make_tuple(Sequence<>{},
                                   Sequence<>{},
                                   Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<6>{},
                                   Sequence<4>{},
                                   Sequence<5>{}));
                }
#endif
            }
        }();

        return a_wave_desc;
    }

    template <typename BBlockDesc_>
    __host__ __device__ static constexpr auto MakeBWaveDescriptor(const BBlockDesc_&)
    {
        constexpr auto b_wave_desc = [&]() {
            if constexpr(BEnableLds)
            {
                // KBlock_BK0_NBlock_NPerBlock_BK1 ->
                // 0       1       2      3       4       5     6
                // KRepeat-NRepeat-NWaves-KThread-NPerXDL-NPack-KPack
                constexpr auto b_wave_desc_temp = transform_tensor_descriptor(
                    BBlockDesc_{},
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_unmerge_transform(make_tuple(
                                   Number<KRepeat>{}, Number<KThread>{}, Number<K0PerXDL>{})),
                               make_unmerge_transform(make_tuple(Number<NRepeat>{},
                                                                 Number<NWaves>{},
                                                                 Number<NPerXDL>{},
                                                                 Number<NPack>{})),
                               make_pass_through_transform(Number<BK1>{})),
                    make_tuple(
                        Sequence<0>{}, Sequence<2>{}, Sequence<1>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<>{},
                               Sequence<>{},
                               Sequence<0, 3, 6>{},
                               Sequence<1, 2, 4, 5>{},
                               Sequence<7>{}));

                return transform_tensor_descriptor(
                    b_wave_desc_temp,
                    make_tuple(make_pass_through_transform(b_wave_desc_temp.GetLength(I0)),
                               make_pass_through_transform(b_wave_desc_temp.GetLength(I1)),
                               make_pass_through_transform(b_wave_desc_temp.GetLength(I2)),
                               make_pass_through_transform(b_wave_desc_temp.GetLength(I3)),
                               make_pass_through_transform(b_wave_desc_temp.GetLength(I4)),
                               make_pass_through_transform(b_wave_desc_temp.GetLength(I5)),
                               make_merge_transform(make_tuple(b_wave_desc_temp.GetLength(I6),
                                                               b_wave_desc_temp.GetLength(I7)))),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6, 7>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6>{}));
            }
            else
            {
#if 0
                return transform_tensor_descriptor(
                    BBlockDesc_{},
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_pass_through_transform(BBlockDesc_{}.GetLength(I1)),
                               make_pass_through_transform(BBlockDesc_{}.GetLength(I3)),
                               make_pass_through_transform(BBlockDesc_{}.GetLength(I4)),
                               make_pass_through_transform(BBlockDesc_{}.GetLength(I5)),
                               make_pass_through_transform(BBlockDesc_{}.GetLength(I6)),
                               make_merge_transform(make_tuple(BBlockDesc_{}.GetLength(I7),
                                                    BBlockDesc_{}.GetLength(I8)))),
                    make_tuple(Sequence<0>{},
                               Sequence<2>{},
                               Sequence<1>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6>{},
                               Sequence<7, 8>{}),
                    make_tuple(Sequence<>{},
                               Sequence<>{},
                               Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}));
#endif
#if 1
                if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
                {
                    // KBlock->KRepeat->NBlock->NRepeat->NWaves->KThread->NPerXDL->K0PerXDL->K1
                    // ->
                    // KRepeat-NRepeat-NWaves-KThread-NPerXDL-NPack-KPack
                    return make_naive_tensor_descriptor_packed(
                        make_tuple(BBlockDesc_{}.GetLength(I1),
                                   BBlockDesc_{}.GetLength(I3),
                                   BBlockDesc_{}.GetLength(I4),
                                   BBlockDesc_{}.GetLength(I5),
                                   BBlockDesc_{}.GetLength(I6),
                                   Number<NPack>{},
                                   BBlockDesc_{}.GetLength(I7) * BBlockDesc_{}.GetLength(I8)));
                }
                else
                {
                    // KBlock - KXDL - NBlock - NRepeat - NWaves - KThread - KPack - NPerXDL - NPack
                    // ->
                    // KXDL - NRepeat - NWaves - KThread- NPerXDL- NPack - KPack
                    return transform_tensor_descriptor(
                        BBlockDesc_{},
                        make_tuple(make_freeze_transform(I0),
                                   make_freeze_transform(I0),
                                   make_pass_through_transform(BBlockDesc_{}.GetLength(I1)),
                                   make_pass_through_transform(BBlockDesc_{}.GetLength(I3)),
                                   make_pass_through_transform(BBlockDesc_{}.GetLength(I4)),
                                   make_pass_through_transform(BBlockDesc_{}.GetLength(I5)),
                                   make_pass_through_transform(BBlockDesc_{}.GetLength(I6)),
                                   make_pass_through_transform(BBlockDesc_{}.GetLength(I7)),
                                   make_pass_through_transform(BBlockDesc_{}.GetLength(I8))),
                        make_tuple(Sequence<0>{},
                                   Sequence<2>{},
                                   Sequence<1>{},
                                   Sequence<3>{},
                                   Sequence<4>{},
                                   Sequence<5>{},
                                   Sequence<6>{},
                                   Sequence<7>{},
                                   Sequence<8>{}),
                        make_tuple(Sequence<>{},
                                   Sequence<>{},
                                   Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<6>{},
                                   Sequence<4>{},
                                   Sequence<5>{}));
                }
#endif
            }
        }();

        return b_wave_desc;
    }

    __host__ __device__ static constexpr auto MakeABlockSliceCopyStep()
    {
        constexpr auto a_block_copy_step = [&]() {
            if constexpr(AEnableLds)
            {
                return make_multi_index(0, K0PerBlock, 0, 0, 0);
            }
            else
            {
                return make_multi_index(0, KRepeat, 0, 0, 0, 0, 0, 0, 0);
            }
        }();

        return a_block_copy_step;
    }

    __host__ __device__ static constexpr auto MakeBBlockSliceCopyStep()
    {
        constexpr auto b_block_copy_step = [&]() {
            if constexpr(BEnableLds)
            {
                return make_multi_index(0, K0PerBlock, 0, 0, 0);
            }
            else
            {
                return make_multi_index(0, KRepeat, 0, 0, 0, 0, 0, 0, 0);
            }
        }();

        return b_block_copy_step;
    }

    __host__ __device__ static constexpr bool CheckValidity(const Argument& karg)
    {
        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(karg.M % MPerBlock == 0))
            {
#if DEBUG_LOG
                std::cout << "Arg M value is not a multiple of MPerBlock! M: " << karg.M << " "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(karg.N % NPerBlock == 0))
            {
#if DEBUG_LOG
                std::cout << "Arg N value is not a multiple of NPerBlock! N: " << karg.N << " "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {

            auto K_t = karg.k_batch * K0PerBlock * K1;
            if(!(karg.K % K_t == 0))
            {
#if DEBUG_LOG
                std::cout << "Arg K value is not a multiple of K_Batch * K0PerBlock * K1! K: "
                          << karg.K << " " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(karg.K % ABlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg K (" << karg.K
                          << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                          << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            if(karg.M % ABlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg M (" << karg.M
                          << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                          << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(karg.N % BBlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg N (" << karg.N
                          << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                          << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            if(karg.K % BBlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg K (" << karg.K
                          << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                          << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(karg.N % CBlockTransferScalarPerVector_NWaveNPerXDL != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg N (" << karg.N
                          << ") value is not a multiple of "
                             "CBlockTransferScalarPerVector_NWaveNPerXDL ("
                          << CBlockTransferScalarPerVector_NWaveNPerXDL << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            if(karg.M % CBlockTransferScalarPerVector_NWaveNPerXDL != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg M (" << karg.M
                          << ") value is not a multiple of "
                             "CBlockTransferScalarPerVector_NWaveNPerXDL ("
                          << CBlockTransferScalarPerVector_NWaveNPerXDL << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        const auto num_k_loop = karg.K0Padded / K0PerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
#if DEBUG_LOG
            std::cout << "The number of k loops (" << num_k_loop
                      << ") value is not supported by GridwiseGemm Pipeline."
                      << " K0Padded: " << karg.K0Padded << ", K0PerBlock: " << K0PerBlock << " "
                      << __FILE__ << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
#endif // DEBUG_LOG
            return false;
        }

        return true;
    }

    __host__ __device__ static auto GetKPad(index_t K, index_t KBlock)
    {
        const index_t K0Padded =
            math::integer_divide_ceil(K, K1 * K0PerBlock * KBlock) * K0PerBlock;
        const index_t KPad = KBlock * K0Padded * K1;
        return KPad;
    }

    __host__ __device__ static constexpr bool CalculateHasMainK0BlockLoop(index_t K0Padded)
    {
        const index_t num_loop = K0Padded / K0PerBlock;
        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    template <typename CGridDesc>
    __host__ __device__ static constexpr auto
    MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        return transform_tensor_descriptor(
            c_m_n_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    template <typename CGridDesc>
    __host__ __device__ static constexpr auto MakeCBlockClusterAdaptor(
        const CGridDesc& c_m_n_grid_desc, index_t /* M01 */, index_t /* N01 */, index_t KBlock)
    {
        return BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc>(
            c_m_n_grid_desc, 8, KBlock);
    }

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        return make_naive_tensor_descriptor_packed(
            make_tuple(I1,
                       Number<CShuffleMRepeatPerShuffle * MWaves * MPerXDL * MPack>{},
                       I1,
                       Number<CShuffleNRepeatPerShuffle * NWaves * NPerXDL * NPack>{}));
    }

    // return block_id to C matrix tile idx (m0, n0, k_split) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap()
    {
        return BlockToCTileMap_3DGrid_KSplit<MPerBlock, NPerBlock>();
    }

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment

        static constexpr auto max_lds_align = K1;

        static constexpr auto a_block_space_size_aligned =
            AEnableLds ? math::integer_least_multiple(MakeABlockDescriptor().GetElementSpaceSize(),
                                                      max_lds_align)
                       : 0;
        static constexpr auto b_block_space_size_aligned =
            BEnableLds ? math::integer_least_multiple(MakeBBlockDescriptor().GetElementSpaceSize(),
                                                      max_lds_align)
                       : 0;

        static constexpr auto a_block_space_offset = 0;
        static constexpr auto b_block_space_offset = a_block_space_size_aligned;

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_space_size =
            GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock().GetElementSpaceSize();

        static constexpr auto c_shuffle_block_space_offset = 0;

        static constexpr auto lds_size =
            math::max(c_shuffle_block_space_size * sizeof(FloatC),
                      a_block_space_size_aligned * sizeof(ComputeType) +
                          b_block_space_size_aligned * sizeof(ComputeType));
    };

    using CGridDesc_M_N         = remove_cvref_t<decltype(MakeCGridDescriptor_M_N(1, 1, 1))>;
    using DefaultBlock2CTileMap = remove_cvref_t<decltype(MakeDefaultBlock2CTileMap())>;

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              typename Block2CTileMap>
    __device__ static void Run(const Argument& karg,
                               void* __restrict__ p_shared_block,
                               const Block2CTileMap& block_2_ctile_map,
                               const AElementwiseOperation a_element_op = AElementwiseOperation{},
                               const BElementwiseOperation b_element_op = BElementwiseOperation{},
                               const CElementwiseOperation c_element_op = CElementwiseOperation{})
    {
        const FloatA* p_a_grid = karg.p_a_grid;
        const FloatB* p_b_grid = karg.p_b_grid;
        FloatC* p_c_grid       = karg.p_c_grid;
        const auto a_grid_desc = MakeAGridDescriptor(
            karg.M, karg.MPadded, karg.K, karg.StrideA, karg.k_batch, karg.K0Padded, karg.KPadded);
        const auto b_grid_desc = MakeBGridDescriptor(
            karg.K, karg.NPadded, karg.N, karg.StrideB, karg.k_batch, karg.K0Padded, karg.KPadded);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(karg.M, karg.N, karg.StrideC);

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n);

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        // divide block work by [KBlock, M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I2]);
        const index_t k_batch_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);

        // A matrix in LDS/VGPR memory, dst of blockwise copy
        constexpr auto a_block_desc = MakeABlockDescriptor();

        // B matrix in LDS/VGPR memory, dst of blockwise copy
        constexpr auto b_block_desc = MakeBBlockDescriptor();

        auto a_block_trait = [&]() {
            // A matrix blockwise copy
            if constexpr(AEnableLds)
            {
                ComputeType* p_a_block = static_cast<ComputeType*>(p_shared_block);
                auto a_block_buf       = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    p_a_block, SharedMemTrait::a_block_space_size_aligned);

                // A matrix blockwise copy
                auto a_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    AElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<1, K0PerBlock, 1, MPerBlock, K1>,
                    ABlockTransferThreadClusterLengths_K0_M_K1,
                    ABlockTransferThreadClusterArrangeOrder,
                    FloatA,
                    ComputeType,
                    decltype(a_grid_desc),
                    decltype(a_block_desc),
                    ABlockTransferSrcAccessOrder,
                    Sequence<0, 3, 1, 2, 4>,
                    ABlockTransferSrcVectorDim,
                    4,
                    ABlockTransferSrcScalarPerVector,
                    ABlockTransferDstScalarPerVector_K1,
                    1,
                    1,
                    AThreadTransferSrcResetCoordinateAfterRun,
                    true>(a_grid_desc,
                          make_multi_index(k_batch_id, 0, block_m_id, 0, 0),
                          a_element_op,
                          a_block_desc,
                          make_multi_index(0, 0, 0, 0, 0),
                          ck::tensor_operation::element_wise::PassThrough{});

                return make_tuple(a_block_buf, a_blockwise_copy);
            }
            else
            {
                auto a_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeType>(
                    a_block_desc.GetElementSpaceSize());

                auto a_blockwise_copy = [&]() {
                    if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
                    {
                        // 0          1          2         3          4          5           6 7 8
                        // [KBlock] - KRepeat - [MBlock] - MRepeat - [MWaves] - [KThread] -
                        // [MPerXDL] - A_K0PerXDL - A_K1
                        return ThreadwiseTensorSliceTransfer_v2<
                            ComputeType,
                            ComputeType,
                            decltype(a_grid_desc),
                            decltype(a_block_desc),
                            Sequence<I1,
                                     Number<KRepeat>{},
                                     I1,
                                     Number<MRepeat>{},
                                     I1,
                                     I1,
                                     I1,
                                     Number<K0PerXDL>{},
                                     Number<K1Value>{}>,
                            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>,
                            8,
                            ABlockTransferSrcScalarPerVector,
                            AThreadTransferSrcResetCoordinateAfterRun,
                            true>(a_grid_desc,
                                  make_multi_index(k_batch_id,
                                                   0,
                                                   block_m_id,
                                                   0,
                                                   get_warp_local_1d_id(),
                                                   (get_thread_local_1d_id() % 64) / MPerXDL,
                                                   get_thread_local_1d_id() % MPerXDL,
                                                   0,
                                                   0));
                    }
                    else
                    {
                        // KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - KPack - MPerXDL -
                        // MPack
                        return ThreadwiseTensorSliceTransfer_v2<
                            ComputeType,
                            ComputeType,
                            decltype(a_grid_desc),
                            decltype(a_block_desc),
                            Sequence<I1,
                                     Number<KRepeat>{},
                                     I1,
                                     Number<MRepeat>{},
                                     I1,
                                     I1,
                                     Number<KPack>{},
                                     I1,
                                     Number<MPack>{}>,
                            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>,
                            8,
                            ABlockTransferSrcScalarPerVector,
                            AThreadTransferSrcResetCoordinateAfterRun,
                            true>(a_grid_desc,
                                  make_multi_index(k_batch_id,
                                                   0,
                                                   block_m_id,
                                                   0,
                                                   get_warp_local_1d_id(),
                                                   (get_thread_local_1d_id() % 64) / MPerXDL,
                                                   0,
                                                   get_thread_local_1d_id() % MPerXDL,
                                                   0));
                    }
                }();

                return make_tuple(a_block_buf, a_blockwise_copy);
            }
        };

        auto b_block_trait = [&]() {
            if constexpr(BEnableLds)
            {
                ComputeType* p_b_block = static_cast<ComputeType*>(p_shared_block) +
                                         SharedMemTrait::b_block_space_offset;
                auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    p_b_block, SharedMemTrait::b_block_space_size_aligned);

                // B matrix blockwise copy
                auto b_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    BElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<1, K0PerBlock, 1, NPerBlock, K1>,
                    BBlockTransferThreadClusterLengths_K0_N_K1,
                    BBlockTransferThreadClusterArrangeOrder,
                    FloatB,
                    ComputeType,
                    decltype(b_grid_desc),
                    decltype(b_block_desc),
                    BBlockTransferSrcAccessOrder,
                    Sequence<0, 3, 1, 2, 4>,
                    BBlockTransferSrcVectorDim,
                    4,
                    BBlockTransferSrcScalarPerVector,
                    BBlockTransferDstScalarPerVector_K1,
                    1,
                    1,
                    BThreadTransferSrcResetCoordinateAfterRun,
                    true>(b_grid_desc,
                          make_multi_index(k_batch_id, 0, block_n_id, 0, 0),
                          b_element_op,
                          b_block_desc,
                          make_multi_index(0, 0, 0, 0, 0),
                          ck::tensor_operation::element_wise::PassThrough{});

                return make_tuple(b_block_buf, b_blockwise_copy);
            }
            else
            {

                auto b_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeType>(
                    b_block_desc.GetElementSpaceSize());

                auto b_blockwise_copy = [&]() {
                    if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
                    {
                        // KBlock - KRepeat - NBlock - NRepeat - NWaves - KThread - NPerXDL -
                        // BK0PerXDL - B_K1
                        return ThreadwiseTensorSliceTransfer_v2<
                            ComputeType,
                            ComputeType,
                            decltype(b_grid_desc),
                            decltype(b_block_desc),
                            Sequence<I1,
                                     Number<KRepeat>{},
                                     I1,
                                     Number<NRepeat>{},
                                     I1,
                                     I1,
                                     I1,
                                     Number<K0PerXDL>{},
                                     Number<K1Value>{}>,
                            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>,
                            8,
                            BBlockTransferSrcScalarPerVector,
                            BThreadTransferSrcResetCoordinateAfterRun,
                            true>(b_grid_desc,
                                  make_multi_index(k_batch_id,
                                                   0,
                                                   block_n_id,
                                                   0,
                                                   get_warp_local_1d_id(),
                                                   (get_thread_local_1d_id() % 64) / NPerXDL,
                                                   get_thread_local_1d_id() % NPerXDL,
                                                   0,
                                                   0));
                    }
                    else
                    {
                        // KBlock - KXDL - MBlock - MRepeat - MWaves - KThread - KPack - MPerXDL -
                        // MPack
                        return ThreadwiseTensorSliceTransfer_v2<
                            ComputeType,
                            ComputeType,
                            decltype(b_grid_desc),
                            decltype(b_block_desc),
                            Sequence<I1,
                                     Number<KRepeat>{},
                                     I1,
                                     Number<NRepeat>{},
                                     I1,
                                     I1,
                                     Number<KPack>{},
                                     I1,
                                     Number<NPack>{}>,
                            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>,
                            8,
                            BBlockTransferSrcScalarPerVector,
                            BThreadTransferSrcResetCoordinateAfterRun,
                            true>(b_grid_desc,
                                  make_multi_index(k_batch_id,
                                                   0,
                                                   block_n_id,
                                                   0,
                                                   get_warp_local_1d_id(),
                                                   (get_thread_local_1d_id() % 64) / NPerXDL,
                                                   0,
                                                   get_thread_local_1d_id() % NPerXDL,
                                                   0));
                    }
                }();

                return make_tuple(b_block_buf, b_blockwise_copy);
            }
        };

        auto a_block_buf      = a_block_trait()[I0];
        auto a_blockwise_copy = a_block_trait()[I1];

        auto b_block_buf      = b_block_trait()[I0];
        auto b_blockwise_copy = b_block_trait()[I1];

        constexpr auto a_block_slice_copy_step = MakeABlockSliceCopyStep();
        constexpr auto b_block_slice_copy_step = MakeBBlockSliceCopyStep();

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check

        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            ComputeType, // ComputeType A
            ComputeType, // ComputeType B
            FloatAcc,
            decltype(MakeAWaveDescriptor(a_block_desc)),
            decltype(MakeBWaveDescriptor(b_block_desc)),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXDL,
            NPerXDL,
            MRepeat,
            NRepeat,
            KPack,
            LoopSched,
            AEnableLds,
            BEnableLds,
            MPack,
            NPack>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // gridwise GEMM pipeline
        const auto K = [&]() {
            if constexpr(AEnableLds)
            {
                return a_grid_desc.GetLength(I1) * a_grid_desc.GetLength(I4);
            }
            else
            {
                if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
                {
                    return a_grid_desc.GetLength(I1) * a_grid_desc.GetLength(I7) *
                           a_grid_desc.GetLength(I5) * a_grid_desc.GetLength(I8);
                }
                else
                {
                    return a_grid_desc.GetLength(I1) * a_grid_desc.GetLength(I5) *
                           a_grid_desc.GetLength(I6);
                }
            }
        }();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(K / KPerBlock);

        const auto gridwise_gemm_pipeline = GridwiseGemmPipe{};

        gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(a_grid_desc,
                                                               a_block_desc,
                                                               a_blockwise_copy,
                                                               a_grid_buf,
                                                               a_block_buf,
                                                               a_block_slice_copy_step,
                                                               b_grid_desc,
                                                               b_block_desc,
                                                               b_blockwise_copy,
                                                               b_grid_buf,
                                                               b_block_buf,
                                                               b_block_slice_copy_step,
                                                               blockwise_gemm,
                                                               c_thread_buf,
                                                               num_k_block_main_loop);

        // output: register to global memory
        {
            constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc =
                blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc =
                blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto M0 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I0);
            constexpr auto N0 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I1);
            constexpr auto M1 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I2);
            constexpr auto N1 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I3);
            constexpr auto M2 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I4);
            constexpr auto M3 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I5);
            constexpr auto M4 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I6);
            constexpr auto N2 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I7);

            constexpr auto c_block_desc_mblock_mperblock_nblock_nperblock =
                GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<FloatC*>(p_shared_block),
                c_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                c_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0), // freeze mblock
                    make_unmerge_transform(make_tuple(CShuffleMRepeatPerShuffle,
                                                      M1,
                                                      M2,
                                                      M3,
                                                      M4)), // M1 = MWaves, M2 * M3 * M4 = MPerXDL
                    make_freeze_transform(I0),              // freeze nblock
                    make_unmerge_transform(make_tuple(CShuffleNRepeatPerShuffle,
                                                      N1,
                                                      N2))), // M1 = MWaves, M2 * M3 * M4 = MPerXDL
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4, 5, 6>{}, Sequence<>{}, Sequence<1, 3, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatAcc,
                                                   FloatC,
                                                   decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMRepeatPerShuffle,
                                                            CShuffleNRepeatPerShuffle,
                                                            I1,
                                                            I1,
                                                            M2,
                                                            I1,
                                                            M4,
                                                            I1>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3],
                                     m_thread_data_on_block_idx[I4],
                                     n_thread_data_on_block_idx[I2]),
                    ck::tensor_operation::element_wise::PassThrough{}};

            // LDS to global
            auto c_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // index_t BlockSize,
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMRepeatPerShuffle * MWaves * MPerXDL,
                         1,
                         CShuffleNRepeatPerShuffle * NWaves * NPerXDL>, // BlockSliceLengths,
                CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                FloatC,               // typename SrcData,
                FloatC,               // typename DstData,
                decltype(c_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                       // typename DimAccessOrder,
                3,                                          // index_t VectorDim,
                CBlockTransferScalarPerVector_NWaveNPerXDL, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun
                {c_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_m_id, 0, block_n_id, 0),
                 c_element_op};

            constexpr auto mxdlperwave_forward_step =
                make_multi_index(0, CShuffleMRepeatPerShuffle * MWaves * MPerXDL, 0, 0);
            constexpr auto nxdlperwave_forward_step =
                make_multi_index(0, 0, 0, CShuffleNRepeatPerShuffle * NWaves * NPerXDL);
            constexpr auto nxdlperwave_backward_step =
                make_multi_index(0, 0, 0, -CShuffleNRepeatPerShuffle * NWaves * NPerXDL);

            static_for<0, MRepeat, CShuffleMRepeatPerShuffle>{}([&](auto mxdlperwave_iter) {
                constexpr auto mxdlperwave = mxdlperwave_iter;

                static_for<0, NRepeat, CShuffleNRepeatPerShuffle>{}([&](auto nxdlperwave_iter) {
                    constexpr bool nxdlperwave_forward_sweep =
                        (mxdlperwave % (2 * CShuffleMRepeatPerShuffle) == 0);

                    constexpr index_t nxdlperwave_value =
                        nxdlperwave_forward_sweep
                            ? nxdlperwave_iter
                            : (NRepeat - nxdlperwave_iter - CShuffleNRepeatPerShuffle);

                    constexpr auto nxdlperwave = Number<nxdlperwave_value>{};

                    // make sure it's safe to do ds_write
                    block_sync_lds();

                    // VGPR to LDS
                    c_thread_copy_vgpr_to_lds.Run(
                        c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc,
                        make_tuple(mxdlperwave, nxdlperwave, I0, I0, I0, I0, I0, I0),
                        c_thread_buf,
                        c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                        c_block_buf);

                    // make sure it's safe to do ds_read
                    block_sync_lds();

                    // LDS to global
                    c_block_copy_lds_to_global.Run(c_block_desc_mblock_mperblock_nblock_nperblock,
                                                   c_block_buf,
                                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                   c_grid_buf);

                    // move on nxdlperwave dimension
                    if constexpr(nxdlperwave_forward_sweep &&
                                 (nxdlperwave < NRepeat - CShuffleNRepeatPerShuffle))
                    {
                        c_block_copy_lds_to_global.MoveDstSliceWindow(
                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                            nxdlperwave_forward_step);
                    }
                    else if constexpr((!nxdlperwave_forward_sweep) && (nxdlperwave > 0))
                    {
                        c_block_copy_lds_to_global.MoveDstSliceWindow(
                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                            nxdlperwave_backward_step);
                    }
                });

                // move on mxdlperwave dimension
                if constexpr(mxdlperwave < MRepeat - CShuffleMRepeatPerShuffle)
                {
                    c_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mperblock_nblock_nperblock, mxdlperwave_forward_step);
                }
            });
        }
    }

    static std::string GetTypeString()
    {
        auto str = std::stringstream();

        // clang-format off
        str << "GemmXdlSplitKCShuffle_"
            << getGemmSpecializationString(GemmSpec) << "_"
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
            << "_"
            << "B" << BlockSize << "_"
            << "Vec" << ABlockTransferSrcScalarPerVector << "x"
            << BBlockTransferSrcScalarPerVector << "x"
            << CBlockTransferScalarPerVector_NWaveNPerXDL << "_"
            << MPerBlock << "x"
            << NPerBlock << "x"
            << K0PerBlock << "x"
            << K1 <<"_"
            << "AEnableLds:" << AEnableLds<<"_"
            << "BEnableLds:" << BEnableLds;
        // clang-format on

        return str.str();
    }
};

} // namespace ck
