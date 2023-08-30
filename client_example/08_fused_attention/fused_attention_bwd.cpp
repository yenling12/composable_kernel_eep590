// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#define PRINT_HOST 0
#define USING_MASK 0
#define DIM 64 // DIM should be a multiple of 8.

#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_mha_bwd_qloop.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_dropout.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16   = ck::half_t;
using BF16  = ck::bhalf_t;
using F32   = float;
using U16   = unsigned short;
using INT32 = int32_t;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

using QKVElementOp = PassThrough;
using YElementOp   = PassThrough;

using InputDataType    = F16;
using OutputDataType   = F16;
using GemmDataType     = F16;
using AccDataType      = F32;
using ShuffleDataType  = F32;
using LSEDataType      = F32;
using ZDataType        = U16; // INT32
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;
// When OutputDataType == F32,      CShuffleBlockTransferScalarPerVector_NPerBlock = 4
// When OutputDataType == F16/BF16, CShuffleBlockTransferScalarPerVector_NPerBlock = 8
static constexpr ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock = 8;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
#if USING_MASK
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskUpperTriangleFromTopLeft;
#else
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;
#endif

static constexpr auto TensorSpecQ   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecK   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecV   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecY   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr bool Deterministic = false;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

template <typename TensorQ,
          typename TensorK,
          typename TensorV,
          typename TensorS,
          typename TensorP,
          typename TensorZ,
          typename TensorY,
          typename TensorLSE = TensorP>
void run_attention_fwd_host(const TensorQ& q_g_m_k,
                            const TensorK& k_g_n_k,
                            const TensorV& v_g_n_o,
                            const float alpha,
                            TensorS& s_g_m_n,
                            TensorP& p_g_m_n,
                            TensorY& y_g_m_o,
                            TensorLSE& lse_g_m,
                            TensorP& p_drop_g_m_n,
                            TensorZ& z_g_m_n,
                            ZDataType p_dropout_in_16bits,
                            float rp_dropout)
{
    // S = alpha * Q * K^T
    auto k_g_k_n            = k_g_n_k.Transpose({0, 2, 1});
    auto ref_gemm0          = ReferenceGemm0Instance{};
    auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
    auto ref_gemm0_argument = ref_gemm0.MakeArgument(
        q_g_m_k, k_g_k_n, s_g_m_n, PassThrough{}, PassThrough{}, Scale{alpha});

    ref_gemm0_invoker.Run(ref_gemm0_argument);

    // masking
    auto M          = s_g_m_n.GetLengths()[1];
    auto N          = s_g_m_n.GetLengths()[2];
    const auto mask = DeviceGemmInstance::C0MatrixMask(M, N);
    s_g_m_n.ForEach([&](auto& self, auto idx) {
        if(mask.IsMaskedElement(idx[1], idx[2]))
            self(idx) = -ck::NumericLimits<float>::Infinity();
    });

    // P = Softmax(S)
    auto ref_softmax          = ReferenceSoftmaxInstance{};
    auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
    auto ref_softmax_argument = ref_softmax.MakeArgument(s_g_m_n, p_g_m_n, 1, 0, {2}, &lse_g_m);

    ref_softmax_invoker.Run(ref_softmax_argument);

    // P_dropped
    auto ref_dropout         = ReferenceDropoutInstance{};
    auto ref_dropout_invoker = ref_dropout.MakeInvoker();
    auto ref_dropout_argment =
        ref_dropout.MakeArgument(z_g_m_n, p_g_m_n, p_drop_g_m_n, p_dropout_in_16bits, rp_dropout);
    ref_dropout_invoker.Run(ref_dropout_argment);

    // Y = P_dropout * V
    auto ref_gemm1          = ReferenceGemm1Instance{};
    auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
    auto ref_gemm1_argument = ref_gemm1.MakeArgument(
        p_drop_g_m_n, v_g_n_o, y_g_m_o, PassThrough{}, PassThrough{}, PassThrough{});

    ref_gemm1_invoker.Run(ref_gemm1_argument);
}

int main(int argc, char* argv[])
{
    ck::index_t M  = 512;
    ck::index_t N  = 512;
    ck::index_t K  = DIM;
    ck::index_t O  = DIM;
    ck::index_t G0 = 4;
    ck::index_t G1 = 6;

    bool input_permute  = false;
    bool output_permute = false;

    float p_drop                    = 0.0;
    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;

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
    else if(argc == 13)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M  = std::stoi(argv[4]);
        N  = std::stoi(argv[5]);
        K  = std::stoi(argv[6]);
        O  = std::stoi(argv[7]);
        G0 = std::stoi(argv[8]);
        G1 = std::stoi(argv[9]);

        p_drop = std::stof(argv[10]);

        input_permute  = std::stoi(argv[11]);
        output_permute = std::stoi(argv[12]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 11: M, N, K, O, G0, G1\n");
        printf("arg10: scale (alpha)\n");
        printf("arg11 to 12: input / output permute\n");
        exit(0);
    }

    float p_dropout               = 1 - p_drop;
    ZDataType p_dropout_in_16bits = ZDataType(std::floor(p_dropout * 65535.0));
    float rp_dropout              = 1.0 / p_dropout;
    float alpha                   = 1.f / std::sqrt(K);

    std::cout << "do_verification: " << do_verification << std::endl;
    std::cout << "init_method: " << init_method << std::endl;
    std::cout << "time_kernel: " << time_kernel << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "O: " << O << std::endl;
    std::cout << "G0: " << G0 << std::endl;
    std::cout << "G1: " << G1 << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "input_permute: " << input_permute << std::endl;
    std::cout << "output_permute: " << output_permute << std::endl;
    std::cout << "p_drop: " << p_drop << std::endl;
    std::cout << "seed: " << seed << std::endl;
    std::cout << "offset: " << offset << std::endl;

    const ck::index_t BatchCount = G0 * G1;

    std::vector<ck::index_t> q_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> q_gs_ms_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // Q layout [G0, M, G1, K]
            : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // Q layout [G0, G1, M, K]

    std::vector<ck::index_t> k_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> k_gs_ns_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // K layout [G0, N, G1, K]
            : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // K layout [G0, G1, N, K]

    std::vector<ck::index_t> v_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> v_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // V layout [G0, N, G1, O]
            : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // V layout [G0, G1, N, O]

    std::vector<ck::index_t> y_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> y_gs_ms_os_strides =
        output_permute
            ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // Y layout [G0, M, G1, O]
            : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // Y layout [G0, G1, M, O]

    std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1, M, N};
    std::vector<ck::index_t> z_gs_ms_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * N, N, G1 * N, 1} // Z layout [G0, M, G1, N]
            : std::vector<ck::index_t>{G1 * M * N, M * N, N, 1}; // Z layout [G0, G1, M, N]
    // The softmax stat log-sum-exp (LSE) is used to speed up softmax calculation in backward pass
    // Pi = exp(Si) / sum(exp(S0) + exp(S1) + ...)
    //    = exp(Si) / exp(log(sum(exp() + ...)))
    //    = exp(Si - log(sum(exp() + ...)))
    //               ^^^^^^^^^^^^^^^^^^^^^
    //                       LSE
    std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides{G1 * M, M, 1}; // LSE layout [G0, G1, M]

    Tensor<InputDataType> q_gs_ms_ks(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
    Tensor<InputDataType> k_gs_ns_ks(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
    Tensor<ZDataType> z_gs_ms_ns(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);
    Tensor<InputDataType> v_gs_os_ns(v_gs_os_ns_lengths, v_gs_os_ns_strides);
    Tensor<InputDataType> y_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
    Tensor<InputDataType> ygrad_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
    Tensor<LSEDataType> lse_gs_ms(lse_gs_ms_lengths, lse_gs_ms_strides);

    std::cout << "q_gs_ms_ks: " << q_gs_ms_ks.mDesc << std::endl;
    std::cout << "k_gs_ns_ks: " << k_gs_ns_ks.mDesc << std::endl;
    std::cout << "z_gs_ms_ns: " << z_gs_ms_ns.mDesc << std::endl;
    std::cout << "v_gs_os_ns: " << v_gs_os_ns.mDesc << std::endl;
    std::cout << "y_gs_ms_os: " << y_gs_ms_os.mDesc << std::endl;
    std::cout << "lse_gs_ms_os: " << lse_gs_ms.mDesc << std::endl;

    z_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<InputDataType>{0});
    switch(init_method)
    {
    case 0: break;
    case 1:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-2, 2});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-2, 2});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-2, 2});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-2, 2});
        break;
    case 2:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<InputDataType>{0.0, 1.0});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<InputDataType>{0.0, 1.0});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<InputDataType>{-0.5, 0.5});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_3<InputDataType>{-0.5, 0.5});
        break;
    case 3:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-5, 5});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
        break;
    case 4:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
        break;
    case 5:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Sequential<2>{}); // dy[g0, g1, m, o]
        // dO dot O = [0; 1; 2; ...]
        break;
    case 6:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Sequential<3>{}); // dy[g0, g1, m, o]
        // assume mnko = 256
        // P = softmax(QK) = 0.0039 * ones
        // O = P V = 0.0039 * ones
        // dP = dO V = [0, 1, 2, ...; 0, 1, 2, ...; ...]
        // dO dot O = [127.5; ...]
        // dS = P * (dP - dO dot O)
        //
        break;
    default:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1}); // dy[g0, g1, m, o]
        // assume mnko = 256
        // P = softmax(QK) = 0.0039 * ones
        // O = P V = 0.0039 * ones
        // dP = dO V = ones
        // dS = P * (dP - (dO dot O))
        //    = 0.0039 * ones * (ones - 0.0039*256)
        //    = 0.0039 * ones * (ones - 1)
        //    = 0
    }

    Tensor<InputDataType> q_g_m_k({BatchCount, M, K});
    Tensor<InputDataType> k_g_n_k({BatchCount, N, K});
    Tensor<ZDataType> z_g_m_n({BatchCount, M, N});
    Tensor<InputDataType> v_g_n_o({BatchCount, N, O});
    Tensor<AccDataType> s_g_m_n({BatchCount, M, N});
    Tensor<InputDataType> p_g_m_n({BatchCount, M, N});
    Tensor<InputDataType> p_drop_g_m_n({BatchCount, M, N});
    Tensor<InputDataType> y_g_m_o({BatchCount, M, O});
    Tensor<LSEDataType> lse_g_m({BatchCount, M});

    q_gs_ms_ks.ForEach(
        [&](auto& self, auto idx) { q_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx); });
    k_gs_ns_ks.ForEach(
        [&](auto& self, auto idx) { k_g_n_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx); });
    v_gs_os_ns.ForEach(
        [&](auto& self, auto idx) { v_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx); });

    // qkv gradients have the same descriptor as with qkv
    DeviceMem q_device_buf(sizeof(InputDataType) * q_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem k_device_buf(sizeof(InputDataType) * k_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem z_device_buf(sizeof(ZDataType) * z_gs_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem v_device_buf(sizeof(InputDataType) * v_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem y_device_buf(sizeof(InputDataType) * y_gs_ms_os.mDesc.GetElementSpaceSize());
    DeviceMem lse_device_buf(sizeof(LSEDataType) * lse_gs_ms.mDesc.GetElementSpaceSize());
    DeviceMem qgrad_device_buf(sizeof(OutputDataType) * q_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem kgrad_device_buf(sizeof(OutputDataType) * k_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem vgrad_device_buf(sizeof(OutputDataType) * v_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem ygrad_device_buf(sizeof(InputDataType) * y_gs_ms_os.mDesc.GetElementSpaceSize());

    q_device_buf.ToDevice(q_gs_ms_ks.mData.data());
    k_device_buf.ToDevice(k_gs_ns_ks.mData.data());
    z_device_buf.ToDevice(z_gs_ms_ns.mData.data());
    v_device_buf.ToDevice(v_gs_os_ns.mData.data());
    ygrad_device_buf.ToDevice(ygrad_gs_ms_os.mData.data());

    using DeviceOp =  ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward<2,
                                                                                            1,
                                                                                            1,
                                                                                            1,
                                                                                            1,
                                                                                            InputDataType,
                                                                                            OutputDataType,
                                                                                            ZDataType,
                                                                                            LSEDataType,
                                                                                            ck::Tuple<>,
                                                                                            ck::Tuple<>,
                                                                                            AElementOp,
                                                                                            B0ElementOp,
                                                                                            Acc0ElementOp,
                                                                                            B1ElementOp,
                                                                                            CElementOp,
                                                                                            MaskingSpec>;
    
    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device op instances
    std::cout << "Run all instances and do timing" << std::endl;
    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(q_device_buf.GetDeviceBuffer(),
                                                        k_device_buf.GetDeviceBuffer(),
                                                        nullptr, // set to nullptr
                                                        v_device_buf.GetDeviceBuffer(),
                                                        y_device_buf.GetDeviceBuffer(),
                                                        lse_device_buf.GetDeviceBuffer(),
                                                        ygrad_device_buf.GetDeviceBuffer(),
                                                        qgrad_device_buf.GetDeviceBuffer(),
                                                        kgrad_device_buf.GetDeviceBuffer(),
                                                        vgrad_device_buf.GetDeviceBuffer(),
                                                        {}, // std::array<void*, 1> p_acc0_biases;
                                                        {}, // std::array<void*, 1> p_acc1_biases;
                                                        q_gs_ms_ks_lengths,
                                                        q_gs_ms_ks_strides,
                                                        k_gs_ns_ks_lengths,
                                                        k_gs_ns_ks_strides,
                                                        z_gs_ms_ns_lengths,
                                                        z_gs_ms_ns_strides,
                                                        v_gs_os_ns_lengths,
                                                        v_gs_os_ns_strides,
                                                        y_gs_ms_os_lengths,
                                                        y_gs_ms_os_strides,
                                                        lse_gs_ms_lengths,
                                                        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
                                                        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
                                                        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
                                                        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
                                                        QKVElementOp{},
                                                        QKVElementOp{},
                                                        Scale{alpha},
                                                        QKVElementOp{},
                                                        YElementOp{},
                                                        p_drop,
                                                        std::tuple<unsigned long long, unsigned long long>(seed, offset));


        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {

            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop = (size_t(3) * M * N * K + size_t(2) * M * N * O) * 2 * BatchCount;

            std::size_t num_btype =
                (sizeof(InputDataType) * M * K + sizeof(InputDataType) * K * N +
                 sizeof(InputDataType) * N * O + sizeof(InputDataType) * M * O * size_t(2) +
                 sizeof(OutputDataType) * M * K + sizeof(OutputDataType) * K * N +
                 sizeof(OutputDataType) * N * O) *
                    BatchCount +
                sizeof(LSEDataType) * M * BatchCount;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_id      = i;
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    // run the best instance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(q_device_buf.GetDeviceBuffer(),
                                                        k_device_buf.GetDeviceBuffer(),
                                                        nullptr, // set to nullptr
                                                        v_device_buf.GetDeviceBuffer(),
                                                        y_device_buf.GetDeviceBuffer(),
                                                        lse_device_buf.GetDeviceBuffer(),
                                                        ygrad_device_buf.GetDeviceBuffer(),
                                                        qgrad_device_buf.GetDeviceBuffer(),
                                                        kgrad_device_buf.GetDeviceBuffer(),
                                                        vgrad_device_buf.GetDeviceBuffer(),
                                                        {}, // std::array<void*, 1> p_acc0_biases;
                                                        {}, // std::array<void*, 1> p_acc1_biases;
                                                        q_gs_ms_ks_lengths,
                                                        q_gs_ms_ks_strides,
                                                        k_gs_ns_ks_lengths,
                                                        k_gs_ns_ks_strides,
                                                        z_gs_ms_ns_lengths,
                                                        z_gs_ms_ns_strides,
                                                        v_gs_os_ns_lengths,
                                                        v_gs_os_ns_strides,
                                                        y_gs_ms_os_lengths,
                                                        y_gs_ms_os_strides,
                                                        lse_gs_ms_lengths,
                                                        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
                                                        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
                                                        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
                                                        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
                                                        QKVElementOp{},
                                                        QKVElementOp{},
                                                        Scale{alpha},
                                                        QKVElementOp{},
                                                        YElementOp{},
                                                        p_drop,
                                                        std::tuple<unsigned long long, unsigned long long>(seed, offset));

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
