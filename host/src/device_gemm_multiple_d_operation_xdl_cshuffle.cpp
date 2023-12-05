// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include <cassert>

namespace ck {
namespace host {
namespace device_gemm_multiple_d {

static std::string GetGemmSpec(const std::size_t m,
                               const std::size_t n,
                               const std::size_t k,
                               const std::size_t m_per_block,
                               const std::size_t n_per_block,
                               const std::size_t k_per_block)
{
    std::string spec = "";
    if(integer_divide_ceil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if(integer_divide_ceil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if(integer_divide_ceil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";
    if(spec == "")
        return "ck::tensor_operation::device::GemmSpecialization::Default";

    return "ck::tensor_operation::device::GemmSpecialization::" + spec + "Padding";
}

template <class F>
std::vector<Operation_Xdl_CShuffle> CreateOperationsImpl(F f, Layout ALayout, Layout BLayout)
{
    std::vector<Operation_Xdl_CShuffle> result;

    std::vector<operation::TileDesc> tile_descriptions = {
        // clang-format off
//  Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl| NumGemmK|
//   Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per| Prefetch|
//       |      |      |      |    |    |     |     | Wave| Wave|    Stage|
//       |      |      |      |    |    |     |     |     |     |         |
  {   256,   256,   128,    32,   8,   2,   32,   32,    4,    2,        1},
  {   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,        1},
//   {   256,   128,   256,    32,   8,   2,   32,   32,    2,    4,        1},
//   {   256,   128,   256,    32,   8,   8,   32,   32,    2,    4,        1},
//   {   128,   128,   128,    32,   8,   2,   32,   32,    4,    2,        1},
//   {   128,   128,   128,    32,   8,   8,   32,   32,    4,    2,        1},
//   {   256,   128,   128,    32,   8,   2,   32,   32,    2,    2,        1},
//   {   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,        1},
//   {   128,   128,    64,    32,   8,   2,   32,   32,    2,    2,        1},
//   {   128,   128,    64,    32,   8,   8,   32,   32,    2,    2,        1},
//   {   128,    64,   128,    32,   8,   2,   32,   32,    2,    2,        1},
//   {   128,    64,   128,    32,   8,   8,   32,   32,    2,    2,        1},
//   {   256,   128,    64,    32,   8,   2,   32,   32,    2,    1,        1},
//   {   256,   128,    64,    32,   8,   8,   32,   32,    2,    1,        1},
//   {   256,    64,   128,    32,   8,   2,   32,   32,    1,    2,        1},
//   {   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,        1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc>
        a_block_descriptions_rowmajor =
            {
                // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//   {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
                // clang-format on
            };

    std::vector<operation::BlockTransferDesc> b_block_descriptions_rowmajor = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |
//                |               |               |              |               |               |          |
  {    S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1},
//   {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0},
//   {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1},
//   {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0},
//   {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1},
//   {    S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0},
//   {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1},
//   {    S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0},
//   {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1},
//   {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0},
//   {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1},
//   {    S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0},
//   {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1},
//   {    S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0},
//   {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1},
        // clang-format on
    };

    std::vector<operation::CShuffleDesc> cshuffle_descriptions = {
        // clang-format off
//    CShuffle|    CShuffle|
// MXdlPerWave| NXdlPerWave|
//  PerShuffle|  PerShuffle|
//            |            |
  {          1,           1},
  {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
//   {          1,           1},
        // clang-format on
    };

    std::vector<operation::CBlockTransferDesc> c_block_descriptions = {
        // clang-format off
// CBlockTransferClusterLengths|  CBlockTransfer
//         _MBlock_MWaveMPerXdl| ScalarPerVector
//         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl
//                             |                
  {              S<1, 32, 1, 8>,               8},
  {              S<1, 32, 1, 8>,               8},
//   {              S<1, 32, 1, 8>,               8},
//   {              S<1, 32, 1, 8>,               8},
//   {              S<1, 16, 1, 8>,               8},
//   {              S<1, 16, 1, 8>,               8},
//   {              S<1, 32, 1, 8>,               8},
//   {              S<1, 32, 1, 8>,               8},
//   {              S<1, 32, 1, 4>,               8},
//   {              S<1, 32, 1, 4>,               8},
//   {              S<1, 16, 1, 8>,               8},
//   {              S<1, 16, 1, 8>,               8},
//   {              S<1, 32, 1, 8>,               8},
//   {              S<1, 32, 1, 8>,               8},
//   {              S<1, 32, 1, 8>,               8},
//   {              S<1, 32, 1, 8>,               8},
        // clang-format on
    };

    const auto a_block_descriptions =
        (ALayout == Layout::Row) ? a_block_descriptions_rowmajor : b_block_descriptions_rowmajor;
    const auto b_block_descriptions =
        (BLayout == Layout::Row) ? b_block_descriptions_rowmajor : a_block_descriptions_rowmajor;

    assert(tile_descriptions.size() == a_block_descriptions.size());
    assert(tile_descriptions.size() == b_block_descriptions.size());
    assert(tile_descriptions.size() == cshuffle_descriptions.size());
    assert(tile_descriptions.size() == c_block_descriptions.size());

    for(std::size_t i = 0; i < tile_descriptions.size(); i++)
    {
        Operation_Xdl_CShuffle x;
        x.tile_desc        = tile_descriptions[i];
        x.a_block_transfer = a_block_descriptions[i];
        x.b_block_transfer = b_block_descriptions[i];
        x.cshuffle         = cshuffle_descriptions[i];
        x.c_block_transfer = c_block_descriptions[i];
        auto all           = f(x);
        result.insert(result.end(), all.begin(), all.end());
    }
    return result;
}

static Layout ToLayout(bool Trans) { return Trans ? Layout::Column : Layout::Row; }
std::vector<Operation_Xdl_CShuffle> Operation_Xdl_CShuffle::CreateOperations()
{
    return CreateOperationsImpl([](auto x) -> std::vector<Operation_Xdl_CShuffle> { return {x}; },
                                Layout::Column,
                                Layout::Row);
}
std::vector<Operation_Xdl_CShuffle> Operation_Xdl_CShuffle::CreateOperations(const Problem& prob)
{
    return CreateOperationsImpl(
        [&](Operation_Xdl_CShuffle x) -> std::array<Operation_Xdl_CShuffle, 1> {
            x.A           = TensorDesc{prob.ADataType, ToLayout(prob.TransA)};
            x.B           = TensorDesc{prob.BDataType, ToLayout(prob.TransB)};
            x.E           = TensorDesc{prob.EDataType, ToLayout(prob.TransE)};
            x.Ds          = Transform(prob.DsTrans, prob.DsDataType, [](auto trans, auto dt) {
                return TensorDesc{dt, ToLayout(trans)};
            });
            x.a_elem_op   = prob.AElementOp;
            x.b_elem_op   = prob.BElementOp;
            x.cde_elem_op = prob.CDEElementOp;
            x.gemm_specialization = GetGemmSpec(prob.M,
                                                prob.N,
                                                prob.K,
                                                x.tile_desc.m_per_block,
                                                x.tile_desc.n_per_block,
                                                x.tile_desc.k_per_block);
            return {x};
        },
        ToLayout(prob.TransA),
        ToLayout(prob.TransB));
}

static const char* const DeviceGemmMultipleD_Xdl_CShuffleTemplate =
    R"(// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

struct AlphaBetaAdd
{
    AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, ck::half_t>(
        ck::half_t& e, const float& c, const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * c + beta_ * ck::type_convert<float>(d));
    };

    float alpha_;
    float beta_;
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = F16;
using EDataType        = F16;

using ALayout = Row;
using BLayout = Col;
using DLayout = Row;
using ELayout = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AlphaBetaAdd;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance =
    ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<${LayoutA},
                                                                   ${LayoutB},
                                                                   ck::Tuple<${LayoutDs}>,
                                                                   ${LayoutE},
                                                                   ${ADataType},
                                                                   ${BDataType},
                                                                   ${AccDataType},
                                                                   ${CShuffleDataType},
                                                                   ck::Tuple<${DsDataType}>,
                                                                   ${EDataType},
                                                                   ${AElementwiseOperation},
                                                                   ${BElementwiseOperation},
                                                                   ${CDEElementwiseOperation},
                                                                   ${GemmSpecialization}, 
                                                                   ${NumGemmkPrefetchStage}, 
                                                                   ${BlockSize},
                                                                   ${MPerBlock}, 
                                                                   ${NPerBlock}, 
                                                                   ${KPerBlock}, 
                                                                   ${AK1}, 
                                                                   ${BK1}, 
                                                                   ${MPerXDL}, 
                                                                   ${NPerXDL},
                                                                   ${MXdlPerWave}, 
                                                                   ${NXdlPerWave},
                                                                   ${ABlockTransferThreadClusterLengths_AK0_M_AK1},
                                                                   ${ABlockTransferThreadClusterArrangeOrder}, 
                                                                   ${ABlockTransferSrcAccessOrder},
                                                                   ${ABlockTransferSrcVectorDim}, 
                                                                   ${ABlockTransferSrcScalarPerVector},
                                                                   ${ABlockTransferDstScalarPerVector_AK1}, 
                                                                   ${ABlockLdsExtraM},
                                                                   ${BBlockTransferThreadClusterLengths_BK0_N_BK1},
                                                                   S<1, 0, 2>,
                                                                   S<1, 0, 2>,
                                                                   2,
                                                                   8,
                                                                   ${BBlockTransferDstScalarPerVector_BK1},
                                                                   ${BBlockLdsExtraN},
                                                                   ${CShuffleMXdlPerWavePerShuffle}, 
                                                                   ${CShuffleNXdlPerWavePerShuffle},
                                                                   ${CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock},
                                                                   ${CDEBlockTransferScalarPerVector_NPerBlock}>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideD = 4096;
    ck::index_t StrideE = 4096;

    float alpha = 1.0f;
    float beta  = 1.0f;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 6)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        alpha = std::stof(argv[4]);
        beta  = std::stof(argv[5]);
    }
    else if(argc == 13)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideD = std::stoi(argv[9]);
        StrideE = std::stoi(argv[10]);

        alpha = std::stof(argv[11]);
        beta  = std::stof(argv[12]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE, alpha, "
               "beta\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<DDataType> d_m_n(f_host_tensor_descriptor(M, N, StrideD, DLayout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "d_m_n: " << d_m_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d_m_n.GenerateTensorValue(GeneratorTensor_2<DDataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_m_n.GenerateTensorValue(GeneratorTensor_3<DDataType>{-0.5, 0.5});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DDataType) * d_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    d_device_buf.ToDevice(d_m_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{alpha, beta};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                               b_device_buf.GetDeviceBuffer(),
                               std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, 1>{StrideD},
                               StrideE,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    if(do_verification)
    {
        Tensor<CShuffleDataType> c_m_n({M, N});

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument =
            ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, a_element_op, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_m_n_host_result(m, n), c_m_n(m, n), d_m_n(m, n));
            }
        }

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
    }

    return 0;
})";

Solution Operation_Xdl_CShuffle::ToSolution() const
{
    // name = (str(operation.tile_desc.block_size) + "_" +
    // std::to_string(this->tile_desc.m_per_block) + "_" +
    // std::to_string(this->tile_desc.n_per_block)
    //+ "_" + str(std::to_string(this->tile_desc.k_per_block)) + "_" + str(operation.tile_desc.k1))
    std::unordered_map<std::string, std::string> values = {
        {"name", ToString(this->A.layout)},
        {"LayoutA", ToString(this->A.layout)},
        {"LayoutB", ToString(this->B.layout)},
        {"LayoutDs",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.layout); }))},
        {"LayoutE", ToString(this->E.layout)},
        {"ADataType", ToString(this->A.element)},
        {"BDataType", ToString(this->B.element)},
        {"AccDataType", ToString(this->acc)},
        {"CShuffleDataType", ToString(this->cs_type)},
        {"DsDataType",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.element); }))},
        {"EDataType", ToString(this->E.element)},
        {"AElementwiseOperation", this->a_elem_op},
        {"BElementwiseOperation", this->b_elem_op},
        {"CDEElementwiseOperation", this->cde_elem_op},
        {"GemmSpecialization", this->gemm_specialization},
        {"NumGemmkPrefetchStage", std::to_string(this->tile_desc.num_gemmk_prefetch_stage)},
        {"BlockSize", std::to_string(this->tile_desc.block_size)},
        {"MPerBlock", std::to_string(this->tile_desc.m_per_block)},
        {"NPerBlock", std::to_string(this->tile_desc.n_per_block)},
        {"KPerBlock", std::to_string(this->tile_desc.k_per_block)},
        {"AK1", std::to_string(this->tile_desc.ak1)},
        {"BK1", std::to_string(this->tile_desc.bk1)},
        {"MPerXDL", std::to_string(this->tile_desc.m_per_XDL)},
        {"NPerXDL", std::to_string(this->tile_desc.n_per_XDL)},
        {"MXdlPerWave", std::to_string(this->tile_desc.m_Xdl_per_wave)},
        {"NXdlPerWave", std::to_string(this->tile_desc.n_Xdl_per_wave)},
        {"ABlockTransferThreadClusterLengths_AK0_M_AK1",
         this->a_block_transfer.thread_cluster_length},
        {"ABlockTransferThreadClusterArrangeOrder",
         this->a_block_transfer.thread_cluster_arrange_order},
        {"ABlockTransferSrcAccessOrder", this->a_block_transfer.src_access_order},
        {"ABlockTransferSrcVectorDim", std::to_string(this->a_block_transfer.src_vec_dim)},
        {"ABlockTransferSrcScalarPerVector",
         std::to_string(this->a_block_transfer.src_scalar_per_vector)},
        {"ABlockTransferDstScalarPerVector_AK1",
         std::to_string(this->a_block_transfer.dst_scalar_per_vector_k1)},
        {"ABlockLdsExtraM", std::to_string(this->a_block_transfer.lds_add_extra_dim)},
        {"BBlockTransferThreadClusterLengths_BK0_N_BK1",
         this->b_block_transfer.thread_cluster_length},
        {"BBlockTransferThreadClusterArrangeOrder",
         this->b_block_transfer.thread_cluster_arrange_order},
        {"BBlockTransferSrcAccessOrder", this->b_block_transfer.src_access_order},
        {"BBlockTransferSrcVectorDim", std::to_string(this->b_block_transfer.src_vec_dim)},
        {"BBlockTransferSrcScalarPerVector",
         std::to_string(this->b_block_transfer.src_scalar_per_vector)},
        {"BBlockTransferDstScalarPerVector_BK1",
         std::to_string(this->b_block_transfer.dst_scalar_per_vector_k1)},
        {"BBlockLdsExtraN", std::to_string(this->b_block_transfer.lds_add_extra_dim)},
        {"CShuffleMXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.m_Xdl_per_wave_per_shuffle)},
        {"CShuffleNXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.n_Xdl_per_wave_per_shuffle)},
        {"CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock",
         this->c_block_transfer.cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl},
        {"CDEBlockTransferScalarPerVector_NPerBlock",
         std::to_string(this->c_block_transfer.scalar_per_vector_n_wave_n_per_Xdl)},
    };

    return Solution{InterpolateString(DeviceGemmMultipleD_Xdl_CShuffleTemplate, values),
                    std::move(values)};
}

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
