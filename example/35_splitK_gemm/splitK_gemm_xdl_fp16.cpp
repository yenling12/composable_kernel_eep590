// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/literals.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType   = F16;
using BDataType   = F16;
using AccDataType = F32;
using CDataType   = F16;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmXdlSplitKCShuffle
    < ADataType, BDataType, CDataType, AccDataType, 
      ALayout, BLayout, CLayout,
      AElementOp,  BElementOp,  CElementOp,
      GemmDefault,   256,
      64,    16,     8,  8,
      16,   16,
      1,    1,
      S<1, 8, 32, 1>, S<0, 2, 1, 3>,  S<0, 2, 1, 3>,  3, 8, 8, true,
      S<1, 8, 16, 1>, S<0, 1, 3, 2>,  S<0, 1, 3, 2>,  2, 1, 8, true, 
      1, 1, S<1, 64, 1, 4>, 4,  F16, ck::PipelineVersion::v1>;
#if 0
       < ADataType, BDataType, CDataType, AccDataType, 
          ALayout, BLayout, CLayout,
          AElementOp,  BElementOp,  CElementOp,
          GemmDefault,   256,
          16,    64,     8,  8,
          16,   16,
          1,    1,
          S<1, 8, 16, 1>, S<0, 2, 1, 3>,  S<0, 2, 1, 3>,  3, 8, 8, true,
          S<1, 8, 32, 1>, S<0, 1, 3, 2>,  S<0, 1, 3, 2>,  2, 2, 8, true, 
          1, 1, S<1, 16, 1, 16>, 4,  F16, ck::PipelineVersion::v1>;
#endif

#include "run_splitK_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_splitK_gemm_example(argc, argv); }
