// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

namespace ck {

template <index_t MNXdlPerWave, index_t MNWaves, index_t MNPerXdl, typename TileDesc_K0_MN_K1>
__host__ __device__ static constexpr auto
MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K_V2(const TileDesc_K0_MN_K1&)
{
    constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
    constexpr index_t K1 = TileDesc_K0_MN_K1{}.GetLength(Number<2>{});

    return transform_tensor_descriptor(
        TileDesc_K0_MN_K1{},
        make_tuple(make_merge_transform_v3_division_mod(make_tuple(Number<K0>{}, Number<K1>{})),
                   make_unmerge_transform(
                       make_tuple(Number<MNXdlPerWave>{}, Number<MNWaves>{}, Number<MNPerXdl>{}))),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
        make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
}

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename AWaveDesc,
          typename BWaveDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          index_t MPack   = 1,
          index_t NPack   = 1,
          bool AEnableLds = true,
          bool BEnableLds = true>
struct BlockwiseGemmXdlops
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t WaveSize = get_warp_size();

    static constexpr index_t KRepeat = AWaveDesc{}.GetLength(I0);

    static constexpr auto xdlops_gemm =
        XdlopsGemm<FloatA, MPerXDL, NPerXDL, KPack, FloatB, false, MPack, NPack>{};

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerXDL * MPack);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerXDL * NPack);

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              FloatAcc,
                              MRepeat * NRepeat,
                              xdlops_gemm.GetRegSizePerXdlops(),
                              true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        if constexpr(AEnableLds)
        {
            const auto wave_idx = GetWaveIdx();

            const auto waveId_m = wave_idx[I0];

            const auto xdlops_a_idx = xdlops_gemm.CalculateAThreadOriginDataIndex();

            return make_tuple(0, 0, waveId_m, xdlops_a_idx[I0], xdlops_a_idx[I1], 0, 0);
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0, 0, 0);
        }
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        if constexpr(BEnableLds)
        {
            const auto wave_idx = GetWaveIdx();

            const auto waveId_n = wave_idx[I1];

            const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();

            return make_tuple(0, 0, waveId_n, xdlops_b_idx[I0], xdlops_b_idx[I1], 0, 0);
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0, 0, 0);
        }
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        constexpr auto mrepeat_mwave_mperxdl_mpack_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerXDL, MPack))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2, 3>{}));

        constexpr auto nrepeat_nwave_nperxdl_npack_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerXDL, NPack))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2, 3>{}));

        const index_t c_thread_m = mrepeat_mwave_mperxdl_mpack_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0], 0))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperxdl_npack_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1], 0))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    // Todo: Not used in GEMM, deal with it later.
    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex8D(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk4D(xdlops_i, blk_i);

        return make_tuple(Number<m0>{},
                          Number<n0>{},
                          waveId_m,
                          waveId_n,
                          blk_idx[I0],
                          blk_idx[I1],
                          blk_idx[I2],
                          blk_idx[I3]);
    }

    using Tuple7 = decltype(CalculateAThreadOriginDataIndex());
    __host__ __device__ BlockwiseGemmXdlops(Tuple7 a_origin = CalculateAThreadOriginDataIndex(),
                                            Tuple7 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
        static_assert(AWaveDesc::IsKnownAtCompileTime() && BWaveDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && NPerBlock % (NPerXDL * NRepeat) == 0,
                      "wrong!");
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL * MPack>{},
                                                           Number<NPerXDL * NPack>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_g_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL * MPack>{},
                                                           Number<NPerXDL * NPack>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_block_desc_g_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(
                           make_tuple(M / (MWaves * MPerXDL * MPack), MWaves, MPerXDL * MPack)),
                       make_unmerge_transform(
                           make_tuple(N / (NWaves * NPerXDL * NPack), NWaves, NPerXDL * NPack))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_G_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_G_M_N& c_grid_desc_g_m_n)
    {
        const auto G = c_grid_desc_g_m_n.GetLength(I0);
        const auto M = c_grid_desc_g_m_n.GetLength(I1);
        const auto N = c_grid_desc_g_m_n.GetLength(I2);

        const auto c_grid_desc_g_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_g_m_n,
            make_tuple(make_pass_through_transform(G),
                       make_unmerge_transform(
                           make_tuple(M / (MWaves * MPerXDL * MPack), MWaves, MPerXDL * MPack)),
                       make_unmerge_transform(
                           make_tuple(N / (NWaves * NPerXDL * NPack), NWaves, NPerXDL * NPack))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3, 5>{}, Sequence<2, 4, 6>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_grid_desc_g_m0_n0_m1_n1_m2_n2);
    }

    static constexpr AWaveDesc a_wave_desc;
    static constexpr BWaveDesc b_wave_desc;

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatB>(
            b_thread_desc_.GetElementSpaceSize());
        static_for<0, KRepeat, 1>{}([&](auto ik) {
            static_for<0, MRepeat, 1>{}([&](auto im) {
                // read A
                a_thread_copy_.Run(a_wave_desc,
                                   make_tuple(ik, im, I0, I0, I0, I0, I0),
                                   a_block_buf,
                                   a_thread_desc_,
                                   make_tuple(ik, im, I0, I0, I0, I0, I0),
                                   a_thread_buf);

                static_for<0, NRepeat, 1>{}([&](auto in) {
                    // read B
                    b_thread_copy_.Run(b_wave_desc,
                                       make_tuple(ik, in, I0, I0, I0, I0, I0),
                                       b_block_buf,
                                       b_thread_desc_,
                                       make_tuple(ik, in, I0, I0, I0, I0, I0),
                                       b_thread_buf);

                    vector_type<FloatA, MPack * KPack> a_thread_vec;
                    vector_type<FloatB, NPack * KPack> b_thread_vec;

                    static_for<0, MPack, 1>{}([&](auto iim) {
                        static_for<0, KPack, 1>{}([&](auto iik) {
                            a_thread_vec.template AsType<FloatA>()(Number<iim * KPack + iik>{}) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(ik, im, 0, 0, 0, iim, iik))>{}];
                        });
                    });

                    static_for<0, NPack, 1>{}([&](auto iin) {
                        static_for<0, KPack, 1>{}([&](auto iik) {
                            b_thread_vec.template AsType<FloatB>()(Number<iin * KPack + iik>{}) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(ik, in, 0, 0, 0, iin, iik))>{}];
                        });
                    });

                    using mfma_input_type_a =
                        typename vector_type<FloatA, xdlops_gemm.K1PerXdlops>::type;
                    using mfma_input_type_b =
                        typename vector_type<FloatB, xdlops_gemm.K1PerXdlops>::type;
                    using mfma_output_type_c =
                        typename vector_type<FloatAcc,
                                             xdlops_gemm.GetRegSizePerXdlops() / MPack /
                                                 NPack>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(im, in, 0));

                    xdlops_gemm.template RunV2(
                        a_thread_vec.template AsType<mfma_input_type_a>(),
                        b_thread_vec.template AsType<mfma_input_type_b>(),
                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                            .template AsType<mfma_output_type_c>());
                });
            });
        });
    }

    protected:
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<KRepeat>{}, Number<MRepeat>{}, I1, I1, I1, Number<MPack>{}, Number<KPack>{}));

    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<KRepeat>{}, Number<NRepeat>{}, I1, I1, I1, Number<NPack>{}, Number<KPack>{}));

    // C[M, N, NumRegXdlops]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, xdlops_gemm.GetRegSizePerXdlops()));

    template <bool EnableLds>
    struct AThreadCopySelector;

    template <>
    struct AThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<
            FloatA,
            FloatA,
            decltype(a_wave_desc),
            decltype(a_thread_desc_),
            Sequence<1, 1, 1, 1, 1, Number<MPack>{}, Number<KPack>{}>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            6,
            Number<KPack>{},
            Number<KPack>{}>;
    };

    template <>
    struct AThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic<
            FloatA,
            FloatA,
            decltype(a_wave_desc),
            decltype(a_thread_desc_),
            ck::tensor_operation::element_wise::PassThrough,
            Sequence<1, 1, 1, 1, 1, Number<MPack>{}, Number<KPack>{}>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            6,
            Number<KPack>{}>;
    };

    template <bool EnableLds>
    struct BThreadCopySelector;

    template <>
    struct BThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<
            FloatB,
            FloatB,
            decltype(b_wave_desc),
            decltype(b_thread_desc_),
            Sequence<1, 1, 1, 1, 1, Number<NPack>{}, Number<KPack>{}>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            6,
            Number<KPack>{},
            Number<KPack>{}>;
    };

    template <>
    struct BThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic<
            FloatB,
            FloatB,
            decltype(b_wave_desc),
            decltype(b_thread_desc_),
            ck::tensor_operation::element_wise::PassThrough,
            Sequence<1, 1, 1, 1, 1, Number<NPack>{}, Number<KPack>{}>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            6,
            Number<KPack>{}>;
    };

    typename AThreadCopySelector<AEnableLds>::type a_thread_copy_;
    typename BThreadCopySelector<BEnableLds>::type b_thread_copy_;
};

// Note: To facilitate the inter-wave loop scheduler, we need to explicitly set the macro
// CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING=1 as a few intrinsics are not yet available in
// the latest ROCm release. For unsupported compilers, inter-wave loop scheduler falls back to the
// default loop scheduler which is given by the macro CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING=0
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename AWaveDesc,
          typename BWaveDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          index_t MPack          = 1,
          index_t NPack          = 1,
          bool AEnableLds        = true,
          bool BEnableLds        = true,
          index_t NumMacClusters = CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING_MAC_CLUSTERS>
struct BlockwiseGemmXdlopsInterwave : public BlockwiseGemmXdlops<BlockSize,
                                                                 FloatA,
                                                                 FloatB,
                                                                 FloatAcc,
                                                                 AWaveDesc,
                                                                 BWaveDesc,
                                                                 MPerBlock,
                                                                 NPerBlock,
                                                                 KPerBlock,
                                                                 MPerXDL,
                                                                 NPerXDL,
                                                                 MRepeat,
                                                                 NRepeat,
                                                                 KPack,
                                                                 MPack,
                                                                 NPack>
{
    using Base = BlockwiseGemmXdlops<BlockSize,
                                     FloatA,
                                     FloatB,
                                     FloatAcc,
                                     AWaveDesc,
                                     BWaveDesc,
                                     MPerBlock,
                                     NPerBlock,
                                     KPerBlock,
                                     MPerXDL,
                                     NPerXDL,
                                     MRepeat,
                                     NRepeat,
                                     KPack,
                                     MPack,
                                     NPack>;

#if CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING
    using Base::a_wave_desc;
    using Base::b_wave_desc;
    using Base::c_thread_buf_;
    using Base::c_thread_desc_;
    using Base::CalculateAThreadOriginDataIndex;
    using Base::CalculateBThreadOriginDataIndex;
    using Base::I0;
    using Base::I1;
    using Base::KRepeat;
    using Base::MWaves;
    using Base::NWaves;
    using Base::WaveSize;
    using Base::xdlops_gemm;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t KPerThread = KPerBlock / xdlops_gemm.K0PerXdlops;

    static constexpr index_t KPerInnerLoop = math::max(KPerThread / NumMacClusters, KPack);

    using Tuple7 = decltype(CalculateAThreadOriginDataIndex());
    __host__ __device__
    BlockwiseGemmXdlopsInterwave(Tuple7 a_origin = CalculateAThreadOriginDataIndex(),
                                 Tuple7 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
        static_assert(AWaveDesc::IsKnownAtCompileTime() && BWaveDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && NPerBlock % (NPerXDL * NRepeat) == 0,
                      "wrong!");
    }

    // 2-wave optimized blockwise gemm
    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatB>(
            b_thread_desc_.GetElementSpaceSize());

        static_for<0, KPerThread, KPerInnerLoop>{}([&](auto ik_read) {
            static_for<0, MRepeat, 1>{}([&](auto im) {
                // read A
                a_thread_copy_.Run(
                    a_wave_desc,
                    make_tuple(
                        Number<ik_read * KRepeat / NumMacClusters>{}, im, I0, I0, I0, I0, I0),
                    a_block_buf,
                    a_thread_desc_,
                    make_tuple(
                        Number<ik_read * KRepeat / NumMacClusters>{}, im, I0, I0, I0, I0, I0),
                    a_thread_buf);
            });
            static_for<0, NRepeat, 1>{}([&](auto in) {
                // read B
                b_thread_copy_.Run(
                    b_wave_desc,
                    make_tuple(
                        Number<ik_read * KRepeat / NumMacClusters>{}, in, I0, I0, I0, I0, I0),
                    b_block_buf,
                    b_thread_desc_,
                    make_tuple(
                        Number<ik_read * KRepeat / NumMacClusters>{}, in, I0, I0, I0, I0, I0),
                    b_thread_buf);
            });
            __builtin_amdgcn_sched_barrier(0);
            // NOTE: Synchronize threads in a workgroup at the start of each MAC cluster, but except
            // the first, as we can shorten non-MAC cluster a bit and there's no observable negative
            // impact. The desired effect is waves in a workgroup executing MAC in sync. This avoids
            // some out-of-sync waves hijacking MAC resource from other workgroups and reducing the
            // chance of latency hiding by waiting for the rest of the workgroup at the eventual
            // sync point.
            if constexpr(ik_read.value != 0 || KPerInnerLoop == KPerThread)
            {
                asm volatile("s_barrier" ::);
                __builtin_amdgcn_sched_barrier(0);
            }
            static_for<0, KPerInnerLoop / KPack, 1>{}([&](auto ik_compute) {
                static_for<0, MRepeat, 1>{}([&](auto im) {
                    static_for<0, NRepeat, 1>{}([&](auto in) {
                        vector_type<FloatA, MPack * KPack> a_thread_vec;
                        vector_type<FloatB, NPack * KPack> b_thread_vec;

                        static_for<0, MPack, 1>{}([&](auto iim) {
                            static_for<0, KPack, 1>{}([&](auto iik) {
                                a_thread_vec.template AsType<FloatA>()(
                                    Number<iim * KPack + iik>{}) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(make_tuple(
                                        Number<ik_read * KRepeat / NumMacClusters + ik_compute>{},
                                        im,
                                        0,
                                        0,
                                        0,
                                        iim,
                                        iik))>{}];
                            });
                        });

                        static_for<0, NPack, 1>{}([&](auto iin) {
                            static_for<0, KPack, 1>{}([&](auto iik) {
                                b_thread_vec.template AsType<FloatB>()(
                                    Number<iin * KPack + iik>{}) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(make_tuple(
                                        Number<ik_read * KRepeat / NumMacClusters + ik_compute>{},
                                        in,
                                        0,
                                        0,
                                        0,
                                        iin,
                                        iik))>{}];
                            });
                        });

                        using mfma_input_type_a =
                            typename vector_type<FloatA, xdlops_gemm.K1PerXdlops>::type;
                        using mfma_input_type_b =
                            typename vector_type<FloatB, xdlops_gemm.K1PerXdlops>::type;
                        using mfma_output_type_c =
                            typename vector_type<FloatAcc,
                                                 xdlops_gemm.GetRegSizePerXdlops() / MPack /
                                                     NPack>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(im, in, 0));

                        // The block_sync_lds() here performs double duty:
                        // A) safeguard against data hazard because barrier from blockwise_gemm is
                        // moved here B) reduce VMEM FIFO congestion by applying small delays to
                        // different wavefronts It is performed near the end of MAC cluster to
                        // minimize lgkmcnt penalty
                        if constexpr(ik_read.value == KPerThread - KPerInnerLoop &&
                                     ik_compute.value == KPerInnerLoop - KPack &&
                                     im.value == MRepeat - 1 && in.value == NRepeat - 1)
                        {
                            __builtin_amdgcn_sched_barrier(0);
                            block_sync_lds();
                            __builtin_amdgcn_sched_barrier(0);
                        }

                        // TODO: insert setprio in more precise manner since we
                        // could have more than >1 MFMA instructions in single call
                        xdlops_gemm.template RunV2(
                            a_thread_vec.template AsType<mfma_input_type_a>(),
                            b_thread_vec.template AsType<mfma_input_type_b>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<mfma_output_type_c>());
                        if constexpr(ik_compute.value == 0 && im.value == 0 && in.value == 0)
                        {
                            __builtin_amdgcn_sched_barrier(0);
                            __builtin_amdgcn_s_setprio(1);
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    });
                });
            });
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_barrier(0);
        });
    }

    protected:
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<KRepeat>{}, Number<MRepeat>{}, I1, I1, I1, Number<MPack>{}, Number<KPack>{}));

    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<KRepeat>{}, Number<NRepeat>{}, I1, I1, I1, Number<NPack>{}, Number<KPack>{}));

    template <bool EnableLds>
    struct AThreadCopySelector;

    template <>
    struct AThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<FloatA,
                                                      FloatA,
                                                      decltype(a_wave_desc),
                                                      decltype(a_thread_desc_),
                                                      Sequence<Number<KRepeat / NumMacClusters>{},
                                                               1,
                                                               1,
                                                               1,
                                                               1,
                                                               Number<MPack>{},
                                                               Number<KPack>{}>,
                                                      Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                      6,
                                                      Number<KPack>{},
                                                      Number<KPack>{}>;
    };

    template <>
    struct AThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic<
            FloatA,
            FloatA,
            decltype(a_wave_desc),
            decltype(a_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<Number<KRepeat / NumMacClusters>{},
                     1,
                     1,
                     1,
                     1,
                     Number<MPack>{},
                     Number<KPack>{}>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            6,
            Number<KPack>{}>;
    };

    template <bool EnableLds>
    struct BThreadCopySelector;

    template <>
    struct BThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<FloatB,
                                                      FloatB,
                                                      decltype(b_wave_desc),
                                                      decltype(b_thread_desc_),
                                                      Sequence<Number<KRepeat / NumMacClusters>{},
                                                               1,
                                                               1,
                                                               1,
                                                               1,
                                                               Number<NPack>{},
                                                               Number<KPack>{}>,
                                                      Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                      6,
                                                      Number<KPack>{},
                                                      Number<KPack>{}>;
    };

    template <>
    struct BThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic<
            FloatB,
            FloatB,
            decltype(b_wave_desc),
            decltype(b_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<Number<KRepeat / NumMacClusters>{},
                     1,
                     1,
                     1,
                     1,
                     Number<NPack>{},
                     Number<KPack>{}>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            6,
            Number<KPack>{}>;
    };

    typename AThreadCopySelector<AEnableLds>::type a_thread_copy_;
    typename BThreadCopySelector<BEnableLds>::type b_thread_copy_;

#endif // #if CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING
};

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          LoopScheduler LoopSched,
          bool AEnableLds = true,
          bool BEnableLds = true,
          index_t MPack   = 1,
          index_t NPack   = 1>
constexpr auto BlockwiseGemmXdlops_Selector()
{
    if constexpr(LoopSched == LoopScheduler::Default)
    {
        return BlockwiseGemmXdlops<BlockSize,
                                   FloatA,
                                   FloatB,
                                   FloatAcc,
                                   AK0MK1BlockDesc,
                                   BK0NK1BlockDesc,
                                   MPerBlock,
                                   NPerBlock,
                                   KPerBlock,
                                   MPerXDL,
                                   NPerXDL,
                                   MRepeat,
                                   NRepeat,
                                   KPack,
                                   MPack,
                                   NPack,
                                   AEnableLds,
                                   BEnableLds>{};
    }
    else if constexpr(LoopSched == LoopScheduler::Interwave)
    {
        return BlockwiseGemmXdlopsInterwave<BlockSize,
                                            FloatA,
                                            FloatB,
                                            FloatAcc,
                                            AK0MK1BlockDesc,
                                            BK0NK1BlockDesc,
                                            MPerBlock,
                                            NPerBlock,
                                            KPerBlock,
                                            MPerXDL,
                                            NPerXDL,
                                            MRepeat,
                                            NRepeat,
                                            KPack,
                                            MPack,
                                            NPack,
                                            AEnableLds,
                                            BEnableLds>{};
    }
};

/**
 * @brief Blockwise gemm
 *
 * Supports
 * 1. regular XDL output M2_M3_M4_M2 and transposed XDL output M2_N2_N3_N4
 * 2. decoupled input tile descriptor and mma tile descriptor in order to support both vgpr and LDS
 * source buffer
 * 3. configurable k index starting position and step size after each FMA/XDL instruction
 */

template <
    index_t BlockSize,
    typename FloatAB,
    typename FloatAcc,
    typename ATileDesc,
    typename BTileDesc,
    typename AMmaTileDesc,
    typename BMmaTileDesc,
    index_t MPerBlock,
    index_t NPerBlock,
    index_t KPerBlock,
    index_t MPerXDL,
    index_t NPerXDL,
    index_t MRepeat,
    index_t NRepeat,
    index_t KPack,
    index_t MPack,
    index_t NPack,
    bool TransposeC = false,
    index_t AMmaKStride =
        KPack* XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, FloatAB, TransposeC, MPack, NPack>{}
            .K0PerXdlops,
    index_t BMmaKStride =
        KPack* XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, FloatAB, TransposeC, MPack, NPack>{}
            .K0PerXdlops>
struct BlockwiseGemmXdlops_v2r2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t WaveSize = get_warp_size();

    static constexpr index_t A_K0 = ATileDesc{}.GetLength(I0);
    static constexpr index_t B_K0 = BTileDesc{}.GetLength(I0);
    static constexpr index_t A_K1 = ATileDesc{}.GetLength(I2);
    static constexpr index_t B_K1 = BTileDesc{}.GetLength(I2);

    static constexpr auto xdlops_gemm =
        XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, FloatAB, TransposeC, MPack, NPack>{};

    static constexpr index_t KPerThread = KPerBlock / xdlops_gemm.K0PerXdlops;

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerXDL);

    static_assert(KPerThread % KPack == 0,
                  "Wrong KPack setting; try increasing KPerThread or decreasing KPack");

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              FloatAcc,
                              MRepeat * NRepeat,
                              xdlops_gemm.GetRegSizePerXdlops(),
                              true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];

        const auto xdlops_a_idx = xdlops_gemm.CalculateAThreadOriginDataIndex();

        return make_tuple(0, waveId_m, xdlops_a_idx[I1], KPack * xdlops_a_idx[I0]);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();

        return make_tuple(0, waveId_n, xdlops_b_idx[I1], KPack * xdlops_b_idx[I0]);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        constexpr auto mrepeat_mwave_mperxdl_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperxdl_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperxdl_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperxdl_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex8D(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk4D(xdlops_i, blk_i);

        return make_tuple(
            m0, n0, waveId_m, waveId_n, blk_idx[I0], blk_idx[I1], blk_idx[I2], blk_idx[I3]);
    }

    using Tuple4 = decltype(CalculateAThreadOriginDataIndex());

    __host__ __device__
    BlockwiseGemmXdlops_v2r2(Tuple4 a_origin = CalculateAThreadOriginDataIndex(),
                             Tuple4 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
        static_assert(AMmaTileDesc::IsKnownAtCompileTime() && BMmaTileDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && NPerBlock % (NPerXDL * NRepeat) == 0,
                      "wrong!");
    }

    __host__ __device__ BlockwiseGemmXdlops_v2r2(const BlockwiseGemmXdlops_v2r2& other)
        : a_thread_copy_(other.a_origin), b_thread_copy_(other.b_origin)
    {
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, N, M0, M1, M2));
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_g_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_block_desc_g_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_G_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_G_M_N& c_grid_desc_g_m_n)
    {
        const auto G = c_grid_desc_g_m_n.GetLength(I0);
        const auto M = c_grid_desc_g_m_n.GetLength(I1);
        const auto N = c_grid_desc_g_m_n.GetLength(I2);

        const auto c_grid_desc_g_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_g_m_n,
            make_tuple(make_pass_through_transform(G),
                       make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3, 5>{}, Sequence<2, 4, 6>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_grid_desc_g_m0_n0_m1_n1_m2_n2);
    }

    static constexpr AMmaTileDesc a_block_desc_m0_m1_m2_k;
    static constexpr BMmaTileDesc b_block_desc_n0_n1_n2_k;

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        static_for<0, KPerThread / KPack, 1>{}([&](auto k) { // k=0,1,2 instead of k=0,kpack*1, ...
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                // read A
                a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                   make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                   a_block_buf,
                                   a_thread_desc_,
                                   make_tuple(I0, I0, I0, I0),
                                   a_thread_buf);

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    // read B
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf,
                                       b_thread_desc_,
                                       make_tuple(I0, I0, I0, I0),
                                       b_thread_buf);
                    vector_type<FloatAB, KPack> a_thread_vec;
                    vector_type<FloatAB, KPack> b_thread_vec;

                    static_for<0, KPack, 1>{}([&](auto i) {
                        a_thread_vec.template AsType<FloatAB>()(i) = a_thread_buf
                            [Number<a_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, i))>{}];
                        b_thread_vec.template AsType<FloatAB>()(i) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, i))>{}];
                    });

                    using mfma_input_type =
                        typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;
                    using mfma_output_type =
                        typename vector_type<FloatAcc,
                                             xdlops_gemm.GetRegSizePerXdlops() / MPack /
                                                 NPack>::type;
                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    xdlops_gemm.template RunV2(
                        a_thread_vec.template AsType<mfma_input_type>(),
                        b_thread_vec.template AsType<mfma_input_type>(),
                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                            .template AsType<mfma_output_type>());
                });
            });
        });
    }

    protected:
    // A[M0, M1, M2, KPack]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<KPack>{}));

    // B[N0, N1, N2, KPack]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<KPack>{}));

    // C[M, N, NumRegXdlops]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, xdlops_gemm.GetRegSizePerXdlops()));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
