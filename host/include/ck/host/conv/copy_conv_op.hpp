// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/operation/gemm.hpp"
#include "ck/host/conv/copy_dev_conv.hpp"

namespace ck {
namespace host {
namespace conv {

struct Copy_Operation_Conv
{
    static std::vector<Copy_Operation_Conv> CreateOperations(const std::string& prologue,
                                                             const std::string& epilogue);
    static std::vector<Copy_Operation_Conv> CreateOperations(const Copy_Problem_Conv& prob,
                                                             const std::string& prologue,
                                                             const std::string& epilogue);
    std::size_t NumDim;
    TensorDesc A{};
    TensorDesc B{};
    DataType acc               = DataType::Float;
    DataType cs_type           = DataType::Half;
    std::vector<TensorDesc> Ds = {};
    TensorDesc E{};
    std::string a_elem_op   = PassThrough;
    std::string b_elem_op   = PassThrough;
    std::string cde_elem_op = PassThrough;
    std::string prologue    = "";
    std::string epilogue    = "";
    std::string conv_specialization =
        "ck::tensor_operation::device::ConvolutionForwardSpecialization::Default";
    std::string gemm_specialization = "ck::tensor_operation::device::GemmSpecialization::Default";
    operation::TileDesc tile_desc{};
    operation::BlockTransferDesc a_block_transfer{};
    operation::BlockTransferDesc b_block_transfer{};
    operation::CShuffleDesc cshuffle{};
    operation::CBlockTransferDesc c_block_transfer{};

    void update_prologue(const std::string& prologue);
    void update_epilogue(const std::string& epilogue);
    Solution ToSolution() const;
};

} // namespace conv
} // namespace host
} // namespace ck