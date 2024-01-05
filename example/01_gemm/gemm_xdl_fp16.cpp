// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_waveletmodel_cshuffle.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using F16 = ck::half_t;
using F32 = float;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmFactory = 
    std::tuple<
#if 0
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Row,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 256, 
        32, 8, 8,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<4, 64, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Row,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 256, 
        32, 8, 2,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<4, 64, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 2, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Col,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 256, 
        32, 8, 8,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,  
        2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Col,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 8,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,  
        2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Row,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 256, 
        32, 8, 8,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<4, 64, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v2>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Row,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 256, 
        32, 8, 2,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<4, 64, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 2, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v2>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Col,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 256, 
        32, 8, 8,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,  
        2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v2>,
#endif
#if 0
    // interwave best:
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Row,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 128, 
        32, 8, 2,
        32,   32,
        4,    2, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 2, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Interwave, ck::PipelineVersion::v1>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Row,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 128, 
        32, 8, 2,
        32,   32,
        4,    2, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 2, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Interwave, ck::PipelineVersion::v1>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Row,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 128, 
        32, 8, 4,
        32,   32,
        4,    2, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 4, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Interwave, ck::PipelineVersion::v1>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Row,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 128, 
        32, 8, 4,
        32,   32,
        4,    2, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 4, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Interwave, ck::PipelineVersion::v1>,
        
    ck::tensor_operation::device::DeviceGemm_Xdl_WaveletModel_CShuffle<
        ALayout, BLayout, CLayout, 
        ADataType, BDataType, AccDataType,              F16, CDataType,  
        AElementOp,  BElementOp,  CElementOp,    GemmDefault,        
        1, 256,  256,   256,   128,    
        32, 8, 2,
        32,   32,
        4,    2, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 2, 0,
        1, 1, S<1, 32, 1, 8>, 8>,
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Row,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        1,   256,
        256, 256, 
        32, 8, 4,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<4, 64, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 4, 4, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        Row,   Col,  Row,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 8,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>
#endif
   ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        ALayout,   BLayout,  CLayout,  
        F16,   F16,  F16,  F32,  F16,
        PassThrough, PassThrough, PassThrough, GemmDefault,
        2,   256,
        256, 256,
        32, 8, 4,
        32,   32,
        4,    4,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>,
        1, 8, 4, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>
    >;

// clang-format on

// using DeviceGemmInstance = DeviceGemmFactory;

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
