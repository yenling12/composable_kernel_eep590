// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;
using I8  = int8_t;
using I32 = int32_t;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdDataDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

static constexpr auto ConvBwdData1x1S1P0 =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0;

template <index_t NDSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename DsDatatype,
          typename CDEElementOp,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wmma_f16_instances = std::tuple<
    // clang-format off
        //########################################|    NumDim|       A|       B|       Ds|       E| AData| BData|    AccData|  CShuffle|      DsData| EData|           A|           B|          CDE|    ConvForward| Block|  MPer|  NPer| K0Per| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|   Spatial|  Layout|  Layout|   Layout|  Layout|  Type|  Type|       Type|  DataType|        Type|  Type| Elementwise| Elementwise|  Elementwise| Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|          |        |        |         |        |      |      |           |          |            |      |   Operation|   Operation|    Operation|               |      |      |      |      |   |      |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|          |        |        |         |        |      |      |           |          |            |      |            |            |             |               |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // generic instance
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,    64,    64,     4,  8,    16,   16,       2,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 32, 1, 4>,               1>,
        // blocksize=256
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 256,   128,   256,     8,  8,    16,   16,       4,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 256,    64,   256,     8,  8,    16,   16,       2,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 256,   128,   256,     8,  8,    16,   16,       4,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 256,   128,    64,     8,  8,    16,   16,       4,       1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        // blocksize=128
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,    64,   128,     8,  8,    16,   16,       2,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,    64,   128,     8,  8,    16,   16,       2,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,   128,    64,     8,  8,    16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,   128,   128,     8,  8,    16,   16,       4,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,    32,   256,     8,  8,    16,   16,       1,       8,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        // blocksize=64
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  64,    32,    64,     8,  8,    16,   16,       1,       4,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,      S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  64,    64,    64,     8,  8,    16,   16,       2,       4,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,      S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  64,    32,    64,     8,  8,    16,   16,       1,       4,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,      S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  64,    32,   128,     8,  8,    16,   16,       1,       8,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,      S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        // blocksize=32
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  32,    16,    64,     8,  8,    16,   16,       1,       4,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,      S<8, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  32,    64,    32,     8,  8,    16,   16,       4,       2,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,      S<8, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  32,    32,    32,     8,  8,    16,   16,       2,       2,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,      S<8, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16,     F32,       F16, Empty_Tuple,   F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  32,    16,    32,     8,  8,    16,   16,       1,       2,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,      S<8, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 16, 1, 2>,               8>
    // clang-format on
    >;

template <index_t NDSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename DsDatatype,
          typename CDEElementOp,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wmma_i8_instances = std::tuple<
    // clang-format off
        //########################################|    NumDim|       A|       B|       Ds|       E| AData| BData|    AccData|  CShuffle|      DsData| EData|           A|           B|          CDE|    ConvForward| Block|  MPer|  NPer| K0Per| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|   Spatial|  Layout|  Layout|   Layout|  Layout|  Type|  Type|       Type|  DataType|        Type|  Type| Elementwise| Elementwise|  Elementwise| Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|          |        |        |         |        |      |      |           |          |            |      |   Operation|   Operation|    Operation|               |      |      |      |      |   |      |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|          |        |        |         |        |      |      |           |          |            |      |            |            |             |               |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // generic instance
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,    64,    64,     4,  16,    16,   16,       2,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,               1,              16,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,            1,              16,         1,           1,           1,               S<1, 32, 1, 4>,               1>,
        // blocksize=256
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 256,    64,   256,     8,  16,    16,   16,       2,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,            8,              16,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        // blocksize=128
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,    64,   256,     8,  16,    16,   16,       2,       8,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,           16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,    64,   128,     8,  16,    16,   16,       2,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,            8,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,   128,   256,     8,  16,    16,   16,       4,       8,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,           16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,    32,   256,     8,  16,    16,   16,       1,       8,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,           16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, 128,   256,   128,     8,  16,    16,   16,       8,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,            8,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,      
        // blocksize=64
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  64,    32,   128,     8,  16,    16,   16,       1,       8,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,      S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,           16,              16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  64,    64,   128,     8,  16,    16,   16,       2,       8,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,      S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,           16,              16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  64,    32,   128,     8,  16,    16,   16,       1,       8,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,      S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,           16,              16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  64,    32,   64,      8,  16,    16,   16,       1,       4,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,      S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,            8,              16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        // blocksize=32
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  32,    16,    64,     8,  16,    16,   16,       1,       4,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,      S<8, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,           16,              16,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  32,    64,    64,     8,  16,    16,   16,       4,       4,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,      S<8, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,           16,              16,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  32,    32,    32,     8,  16,    16,   16,       2,       2,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,      S<8, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,            8,              16,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout, I8,    I8,        I32,        I8, Empty_Tuple,    I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec,  32,    16,    64,     8,  16,    16,   16,       1,       4,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,      S<8, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,           16,              16,         1,           1,           1,               S<1, 16, 1, 2>,               8>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck