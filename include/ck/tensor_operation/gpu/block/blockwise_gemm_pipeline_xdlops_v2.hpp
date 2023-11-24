// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

#if 0
template <index_t MNXdlPerWave, index_t MNWaves, index_t MNPerXdl, typename TileDesc_K0_MN_K1>
__host__ __device__ static constexpr auto
MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K(const TileDesc_K0_MN_K1&)
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
#endif

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
    bool TransposeC = false,
    index_t AMmaKStride =
        KPack* XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, FloatAB, TransposeC>{}.K0PerXdlops,
    index_t BMmaKStride =
        KPack* XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, FloatAB, TransposeC>{}.K0PerXdlops>
struct BlockwiseGemmXdlops_pipeline_v2
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
        XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, FloatAB, TransposeC>{};

    static constexpr index_t KPerThread = KPerBlock / xdlops_gemm.K0PerXdlops;
    static constexpr index_t KRepeat    = KPerThread / KPack;

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
    BlockwiseGemmXdlops_pipeline_v2(Tuple4 a_origin = CalculateAThreadOriginDataIndex(),
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
              typename CThreadBuffer>
    __device__ void Run(const AGridDesc& a_grid_desc,
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
                        CThreadBuffer& c_thread_buf,
                        index_t num_loop) const
    {
        __builtin_amdgcn_sched_barrier(0);
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());
        // Global prefetch 1th
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        // Initialize C
        c_thread_buf.Clear();

        // Write to LDS immediately to save VGPR
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        __builtin_amdgcn_sched_barrier(0);

        // Global prefetch 2th, Hold in VGPR
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // main body
        if constexpr(HasMainLoop)
        {
#if 0
            if(get_thread_local_1d_id() == 0)
            {
                printf("HasMainLoop=True\n");
            }
#endif
            index_t i = 0;

            do
            {
                // Wait all wave produce this K-loop data
                block_sync_lds();
                static_for<0, KRepeat, 1>{}([&](auto k) { // k=1,2 instead of kpack*1, ...
                    /* Read N+1 */
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        // read A
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                           a_block_buf,
                                           a_thread_desc_,
                                           make_tuple(m0, Number<k % 2>{}, I0, I0),
                                           a_thread_buf);
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            // read B
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                               make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                               b_block_buf,
                                               b_thread_desc_,
                                               make_tuple(n0, Number<k % 2>{}, I0, I0),
                                               b_thread_buf);
                        });
                    });
                });

                /* Compute N */
                static_for<0, MRepeat / 2, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, 0, 0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, 0, 0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });

                // schedule 16 ds_read_b128 and 16 v_mfma
                __builtin_amdgcn_sched_group_barrier(0x100, 8, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_barrier(0);
                block_sync_lds();

                // Write 2th prefetch k-loop
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                /* We have entire loop to cover buffer_load latency */

                // Next Global prefetch
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                static_for<MRepeat / 2, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, 0, 0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, 0, 0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I1, 0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I1, 0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });

                // schedule 8 ds_write_b128, 8 buffer_load_dwordx4, 48 v_mfma
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 6, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 6, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 6, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 6, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 6, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 6, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 6, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 6, 0); // MFMA
                __builtin_amdgcn_sched_barrier(0);

                ++i;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            // Wait all wave produce this K-loop data
            block_sync_lds();
            static_for<0, KRepeat, 1>{}([&](auto k) { // k=1,2 instead of kpack*1, ...
                /* Read N+1 */
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    // read A
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(m0, Number<k % 2>{}, I0, I0),
                                       a_thread_buf);

                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        // read B
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf,
                                           b_thread_desc_,
                                           make_tuple(n0, Number<k % 2>{}, I0, I0),
                                           b_thread_buf);
                    });
                });
            });

            /* Compute N */
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    vector_type<FloatAB, KPack> a_thread_vec;
                    vector_type<FloatAB, KPack> b_thread_vec;

                    static_for<0, KPack, 1>{}([&](auto ik) {
                        a_thread_vec.template AsType<FloatAB>()(ik) = a_thread_buf
                            [Number<a_thread_desc_.CalculateOffset(make_tuple(m0, 0, 0, ik))>{}];
                        b_thread_vec.template AsType<FloatAB>()(ik) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(n0, 0, 0, ik))>{}];
                    });

                    using mfma_input_type =
                        typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    xdlops_gemm.template Run(
                        a_thread_vec.template AsType<mfma_input_type>(),
                        b_thread_vec.template AsType<mfma_input_type>(),
                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                });
            });

            // schedule 16 ds_read_b128 and 32 v_mfma
            __builtin_amdgcn_sched_group_barrier(0x100, 8, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 8, 0); // MFMA
            __builtin_amdgcn_sched_barrier(0);
            block_sync_lds();

            // Write 2th prefetch k-loop
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    vector_type<FloatAB, KPack> a_thread_vec;
                    vector_type<FloatAB, KPack> b_thread_vec;

                    static_for<0, KPack, 1>{}([&](auto ik) {
                        a_thread_vec.template AsType<FloatAB>()(ik) = a_thread_buf
                            [Number<a_thread_desc_.CalculateOffset(make_tuple(m0, I1, 0, ik))>{}];
                        b_thread_vec.template AsType<FloatAB>()(ik) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(n0, I1, 0, ik))>{}];
                    });

                    using mfma_input_type =
                        typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    xdlops_gemm.template Run(
                        a_thread_vec.template AsType<mfma_input_type>(),
                        b_thread_vec.template AsType<mfma_input_type>(),
                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                });
            });

            // schedule 8 ds_write_b128, 8 buffer_load_dwordx4, 32 v_mfma
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            __builtin_amdgcn_sched_barrier(0);

            // Wait all wave produce this K-loop data
            block_sync_lds();
            static_for<0, KRepeat, 1>{}([&](auto k) { // k=1,2 instead of kpack*1, ...
                /* Read N+1 */
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    // read A
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(m0, Number<k % 2>{}, I0, I0),
                                       a_thread_buf);
                });

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    // read B
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf,
                                       b_thread_desc_,
                                       make_tuple(n0, Number<k % 2>{}, I0, I0),
                                       b_thread_buf);
                });
            });

            /* Compute N */
            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, k % 2, 0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, k % 2, 0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            // schedule 16 ds_read_b128 and 64 v_mfma
            __builtin_amdgcn_sched_group_barrier(0x100, 8, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 8, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 8, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 8, 0); // MFMA

            __builtin_amdgcn_sched_group_barrier(0x100, 5, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 8, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 8, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x008, 8, 0); // MFMA
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    protected:
    // M1, N1 as double buffer index
    // Read buffer + Compute buffer
    // A[M0, M1, M2, KPack]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{}, I2, I1, Number<KPack>{}));

    // B[N0, N1, N2, KPack]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<NRepeat>{}, I2, I1, Number<KPack>{}));

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
