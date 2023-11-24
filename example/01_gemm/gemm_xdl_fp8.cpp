// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"

using ADataType        = ck::f8_t;
using BDataType        = ck::f8_t;
using CDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = float;

using F8      = ck::f8_t;
using F32     = float;
using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmFactory = std::tuple<
#if 1
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<Row,
                                                          Row,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          F8,
                                                          F32,
                                                          F8,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough,
                                                          GemmDefault,
                                                          1,
                                                          256,
                                                          256,
                                                          128,
                                                          64,
                                                          16,
                                                          4,
                                                          32,
                                                          32,
                                                          4,
                                                          2,
                                                          S<4, 64, 1>,
                                                          S<1, 0, 2>,
                                                          S<1, 0, 2>,
                                                          2,
                                                          16,
                                                          16,
                                                          1,
                                                          S<8, 32, 1>,
                                                          S<0, 2, 1>,
                                                          S<0, 2, 1>,
                                                          1,
                                                          4,
                                                          4,
                                                          0,
                                                          1,
                                                          1,
                                                          S<1, 64, 1, 4>,
                                                          16,
                                                          ck::LoopScheduler::Interwave,
                                                          ck::PipelineVersion::v1>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<Row,
                                                          Row,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          F8,
                                                          F32,
                                                          F8,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough,
                                                          GemmDefault,
                                                          1,
                                                          256,
                                                          256,
                                                          128,
                                                          64,
                                                          16,
                                                          16,
                                                          32,
                                                          32,
                                                          4,
                                                          2,
                                                          S<4, 64, 1>,
                                                          S<1, 0, 2>,
                                                          S<1, 0, 2>,
                                                          2,
                                                          16,
                                                          16,
                                                          1,
                                                          S<4, 64, 1>,
                                                          S<0, 2, 1>,
                                                          S<0, 2, 1>,
                                                          1,
                                                          2,
                                                          16,
                                                          1,
                                                          1,
                                                          1,
                                                          S<1, 64, 1, 4>,
                                                          16,
                                                          ck::LoopScheduler::Interwave,
                                                          ck::PipelineVersion::v1>,
#endif
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<Row,
                                                          Row,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          F8,
                                                          F32,
                                                          F8,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough,
                                                          GemmDefault,
                                                          1,
                                                          256,
                                                          256,
                                                          128,
                                                          64,
                                                          16,
                                                          8,
                                                          32,
                                                          32,
                                                          4,
                                                          2,
                                                          S<4, 64, 1>,
                                                          S<1, 0, 2>,
                                                          S<1, 0, 2>,
                                                          2,
                                                          16,
                                                          16,
                                                          1,
                                                          S<8, 32, 1>,
                                                          S<0, 2, 1>,
                                                          S<0, 2, 1>,
                                                          1,
                                                          4,
                                                          8,
                                                          1,
                                                          1,
                                                          1,
                                                          S<1, 64, 1, 4>,
                                                          16,
                                                          ck::LoopScheduler::Interwave,
                                                          ck::PipelineVersion::v1>>;
using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
