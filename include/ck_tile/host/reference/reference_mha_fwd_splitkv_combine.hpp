// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

#include "ck_tile/host/reference/reference_batched_elementwise.hpp"
#include "ck_tile/host/reference/reference_batched_gemm.hpp"
#include "ck_tile/host/reference/reference_batched_masking.hpp"
#include "ck_tile/host/reference/reference_batched_softmax.hpp"

namespace ck_tile {

template <typename LSEAccTensor,
          typename OutputAccTensor,
          typename LSETensor,
          typename OutputTensor,
          typename OAccElementFunction = identity>
CK_TILE_HOST void reference_mha_fwd_splitkv_combine(
    const LSEAccTensor& lse_acc_nbhs,
    OutputAccTensor output_acc_nbhsd,
    LSETensor& lse_bhs,
    OutputTensor& output_bhsd,
    std::optional<span<const int32_t>> seqstart_q, // only used in group mode
    OAccElementFunction oacc_element_func = {})
{
    const bool is_batch_mode = !seqstart_q.has_value();
    const index_t batch  = (is_batch_mode ? lse_acc_nbhs.get_length(1) : seqstart_q->size() - 1);
    const index_t nhead  = lse_acc_nbhs.get_length(2);
    const index_t hdim_v = output_acc_nbhsd.get_length(4);
    const index_t num_splits = lse_acc_nbhs.get_length(0);

    using LSEAccDataType = tensor_value_t<LSEAccTensor>;
    using LSEDataType    = tensor_value_t<LSETensor>;

    for(index_t i_batch = 0; i_batch < batch; ++i_batch)
    {
        const index_t real_seqlen_q =
            (is_batch_mode ? lse_acc_nbhs.get_length(3)
                           : (*seqstart_q)[i_batch + 1] - (*seqstart_q)[i_batch]);

        // adjust matrix index according to the mode
        const index_t batch_start = (is_batch_mode ? i_batch : 0);
        const index_t batch_end   = batch_start + 1;
        const index_t query_start = (is_batch_mode ? 0 : (*seqstart_q)[i_batch]);
        const index_t query_end   = query_start + real_seqlen_q;

        using Slice = HostTensorSlice;
        // clang-format off
        auto lse_view_hs = lse_bhs
                .index({Slice(0, batch_start, batch_end), Slice(2, query_start, query_end)})
                .squeeze(0);

        auto output_view_hsd = output_bhsd
                .index({Slice(0, batch_start, batch_end), Slice(2, query_start, query_end)})
                .squeeze(0);
        // clang-format on

        ck_tile::HostTensor<LSEAccDataType> lse_logsum_hs({nhead, real_seqlen_q});

        const auto combine = [&](auto i_head) {
            ck_tile::HostTensor<LSEAccDataType> lse_max({real_seqlen_q});
            lse_max.for_each([&](auto& self, auto i) {
                self(i) = -ck_tile::numeric<LSEAccDataType>::infinity();
            });

            for(index_t i_split = 0; i_split < num_splits; ++i_split)
            {
                // clang-format off
                auto lse_acc_view_s = lse_acc_nbhs
                        .index({Slice(0, i_split, i_split + 1),
                                Slice(1, batch_start, batch_end), 
                                Slice(2, i_head, i_head + 1),
                                Slice(3, query_start, query_end)})
                        .squeeze(0)
                        .squeeze(0)
                        .squeeze(0);
                // clang-format on
                for(index_t i_s = 0; i_s < real_seqlen_q; ++i_s)
                {
                    if(lse_max(i_s) < lse_acc_view_s(i_s))
                    {
                        lse_max(i_s) = lse_acc_view_s(i_s);
                    }
                }
            }

            ck_tile::HostTensor<LSEAccDataType> lse_sum({real_seqlen_q});
            lse_sum.for_each([&](auto& self, auto i) { self(i) = 0; });

            for(index_t i_split = 0; i_split < num_splits; ++i_split)
            {
                // clang-format off
                auto lse_acc_view_s = lse_acc_nbhs
                        .index({Slice(0, i_split, i_split + 1),
                                Slice(1, batch_start, batch_end), 
                                Slice(2, i_head, i_head + 1),
                                Slice(3, query_start, query_end)})
                        .squeeze(0)
                        .squeeze(0)
                        .squeeze(0);
                // clang-format on
                for(index_t i_s = 0; i_s < real_seqlen_q; ++i_s)
                {
                    lse_sum(i_s) += ck_tile::exp(lse_acc_view_s(i_s) - lse_max(i_s));
                }
            }

            for(index_t i_s = 0; i_s < real_seqlen_q; ++i_s)
            {
                lse_logsum_hs(i_head, i_s) = ck_tile::log(lse_sum(i_s)) + lse_max(i_s);
                lse_view_hs(i_head, i_s) =
                    ck_tile::type_convert<LSEDataType>(lse_logsum_hs(i_head, i_s));
            }

            {
                // clang-format off
                auto lse_acc_view_s = lse_acc_nbhs
                        .index({Slice(0, /*i_split=*/0, 1),
                                Slice(1, batch_start, batch_end), 
                                Slice(2, i_head, i_head + 1),
                                Slice(3, query_start, query_end)})
                        .squeeze(0)
                        .squeeze(0)
                        .squeeze(0);
                // clang-format on

                for(index_t i_s = 0; i_s < real_seqlen_q; ++i_s)
                {
                    const LSEAccDataType lse_scale =
                        ck_tile::exp(lse_acc_view_s(i_s) - lse_logsum_hs(i_head, i_s));
                    for(index_t i_d = 0; i_d < hdim_v; ++i_d)
                    {
                        output_acc_nbhsd(/*i_split=*/0, i_batch, i_head, i_s, i_d) =
                            lse_scale * output_acc_nbhsd(/*i_split=*/0, i_batch, i_head, i_s, i_d);
                    }
                }
            }

            for(index_t i_split = 1; i_split < num_splits; ++i_split)
            {
                // clang-format off
                auto lse_acc_view_s = lse_acc_nbhs
                        .index({Slice(0, i_split, i_split + 1),
                                Slice(1, batch_start, batch_end), 
                                Slice(2, i_head, i_head + 1),
                                Slice(3, query_start, query_end)})
                        .squeeze(0)
                        .squeeze(0)
                        .squeeze(0);
                // clang-format on

                for(index_t i_s = 0; i_s < real_seqlen_q; ++i_s)
                {
                    const LSEAccDataType lse_scale =
                        ck_tile::exp(lse_acc_view_s(i_s) - lse_logsum_hs(i_head, i_s));
                    for(index_t i_d = 0; i_d < hdim_v; ++i_d)
                    {
                        output_acc_nbhsd(/*i_split=*/0, i_batch, i_head, i_s, i_d) +=
                            lse_scale * output_acc_nbhsd(i_split, i_batch, i_head, i_s, i_d);
                    }
                }
            }

            for(index_t i_s = 0; i_s < real_seqlen_q; ++i_s)
            {
                for(index_t i_d = 0; i_d < hdim_v; ++i_d)
                {
                    output_view_hsd(i_head, i_s, i_d) = oacc_element_func(
                        output_acc_nbhsd(/*i_split=*/0, i_batch, i_head, i_s, i_d));
                }
            }
        };

        make_ParallelTensorFunctor(combine, nhead)(std::thread::hardware_concurrency());
    }
}

} // namespace ck_tile
