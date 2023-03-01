// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = float;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmInstances = std::tuple<
// RRR Gemm AIT
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   256,   
          128, 256, 8, 8, 16, 16, 4, 4,
          S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 32, 1, 8>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   256,   
          256, 128, 8, 8, 16, 16, 8, 2,
          S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 32, 1, 8>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   256,   
          128, 256, 4, 8, 16, 16, 4, 4,
          S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 32, 1, 8>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   256,   
          256, 128, 4, 8, 16, 16, 8, 2,
          S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 32, 1, 8>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   256,   
          128, 128, 8, 8, 16, 16, 4, 2,
          S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 32, 1, 8>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   256,   
          128, 128, 4, 8, 16, 16, 4, 2,
          S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 32, 1, 8>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   256,   
          256, 64, 8, 8, 16, 16, 8, 1,
          S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 32, 1, 8>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   256,   
          64, 256, 8, 8, 16, 16, 2, 4,
          S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 32, 1, 8>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   128,   
          128, 128, 8, 8, 16, 16, 8, 2,
          S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 16, 1, 8>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   128,   
          128, 64, 8, 8, 16, 16, 4, 2,
          S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 32, 1, 4>, 8, 1>,
    ck::tensor_operation::device::DeviceGemmWmma_CShuffle
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  
          AElementOp,  BElementOp,  CElementOp,    GemmDefault,   128,   
          64, 128, 8, 8, 16, 16, 4, 2,
          S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, true,
          1, 1, S<1, 16, 1, 8>, 8, 1>
>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
