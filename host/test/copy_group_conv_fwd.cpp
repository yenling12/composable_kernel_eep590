#include "ck/host/conv/copy_conv_op.hpp"
#include "ck/host/conv/copy_dev_conv.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/types.hpp"
#include "ck/host/utils.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/tensor_operation/gpu/device/impl/copy_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/copy_transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/library/utility/iterator.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/ranges.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <random>
#include <test.hpp>
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <fstream>

// using half = _Float16;
// using half = __fp16;

std::vector<rtc::src_file> get_headers_for_test()
{
    std::vector<rtc::src_file> result;
    auto hs = ck::host::GetHeaders();
    std::transform(
        hs.begin(), hs.end(), std::back_inserter(result), [&](const auto& p) -> rtc::src_file {
            auto s = p.second;
            std::string content{s.first, s.second};
            return {p.first, content};
        });
    return result;
}

template <class T>
rtc::buffer<T> generate_buffer(std::size_t n, std::size_t seed = 0)
{
    rtc::buffer<T> result(n);
    // std::mt19937 gen(seed);
    // std::uniform_real_distribution<double> dis(-1.0);
    // std::generate(result.begin(), result.end(), [&] { return dis(gen); });
    std::fill(result.begin(), result.end(), 1);
    return result;
}

template <class T, class U>
bool allclose(const T& a, const U& b, double atol = 0.01, double rtol = 0.01)
{
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [&](double x, double y) {
        return fabs(x - y) < atol + rtol * fabs(y);
    });
}

std::string classify(double x)
{
    switch(std::fpclassify(x))
    {
    case FP_INFINITE: return "inf";
    case FP_NAN: return "nan";
    case FP_NORMAL: return "normal";
    case FP_SUBNORMAL: return "subnormal";
    case FP_ZERO: return "zero";
    default: return "unknown";
    }
}

template <class Buffer>
void print_classification(const Buffer& x)
{
    std::unordered_set<std::string> result;
    for(const auto& i : x)
        result.insert(classify(i));
    for(const auto& c : result)
        std::cout << c << ", ";
    std::cout << std::endl;
}

template <class Buffer>
void print_statistics(const Buffer& x)
{
    std::cout << "Min value: " << *std::min_element(x.begin(), x.end()) << ", ";
    std::cout << "Max value: " << *std::max_element(x.begin(), x.end()) << ", ";
    double num_elements = x.size();
    auto mean =
        std::accumulate(x.begin(), x.end(), double{0.0}, std::plus<double>{}) / num_elements;
    auto stddev = std::sqrt(
        std::accumulate(x.begin(),
                        x.end(),
                        double{0.0},
                        [&](double r, double v) { return r + std::pow((v - mean), 2.0); }) /
        num_elements);
    std::cout << "Mean: " << mean << ", ";
    std::cout << "StdDev: " << stddev << "\n";
}

template <class Buffer>
void print_preview(const Buffer& x)
{
    if(x.size() <= 10)
    {
        std::for_each(x.begin(), x.end(), [&](double i) { std::cout << i << ", "; });
    }
    else
    {
        std::for_each(x.begin(), x.begin() + 5, [&](double i) { std::cout << i << ", "; });
        std::cout << "..., ";
        std::for_each(x.end() - 5, x.end(), [&](double i) { std::cout << i << ", "; });
    }
    std::cout << std::endl;
}

template <class T>
struct check_all
{
    rtc::buffer<T> data{};
    bool operator()(const rtc::buffer<T>& x)
    {
        if(data.empty())
        {
            data = x;
            return true;
        }
        return allclose(data, x);
    }
};

template <class Solution>
auto report(const Solution& solution, bool pass)
{
    return test::make_predicate(solution.ToTemplateString(), [=] { return pass; });
}

struct Epilogue
{
    Epilogue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename D>
    __host__ __device__ constexpr void operator()(E& e, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>(ck::half_t& e,
                                                                          const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * e + beta_ * ck::type_convert<float>(d));
    }

    float alpha_;
    float beta_;
};

const std::string conv_compile_check = R"__ck__(
#include <${include}>

${template};

)__ck__";

TEST_CASE(test_problem_kernel)
{
    ck::host::conv::Copy_Problem_Conv prob;
    prob.G  = 1;
    prob.N  = 128;
    prob.C  = 192;
    prob.K  = 256;
    prob.Y  = 3;
    prob.X  = 3;
    prob.Hi = 71;
    prob.Wi = 71;
    prob.Ho = 36;
    prob.Wo = 36;
    check_all<ck::half_t> check;
    std::string epilogue = R"(
struct Epilogue
{
    __host__ __device__ Epilogue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename D>
    __host__ __device__ constexpr void operator()(E& e, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>(ck::half_t& e,
                                                                          const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * e + beta_ * ck::type_convert<float>(d));
    }

    float alpha_;
    float beta_;
};
)";
    // std::string epilogue = "";
    std::string prologue = "";

    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};
    static constexpr auto I3 = ck::Number<3>{};

    using CDEElementOp = Epilogue;

    // length+stride arrays
    ck::Array<ck::index_t, 5> in_lengths{static_cast<int>(prob.G),
                                         static_cast<int>(prob.N),
                                         static_cast<int>(prob.C),
                                         static_cast<int>(prob.Hi),
                                         static_cast<int>(prob.Wi)};
    ck::Array<ck::index_t, 5> out_lengths{static_cast<int>(prob.G),
                                          static_cast<int>(prob.N),
                                          static_cast<int>(prob.K),
                                          static_cast<int>(prob.Ho),
                                          static_cast<int>(prob.Wo)};
    ck::Array<ck::index_t, 5> wei_lengths{static_cast<int>(prob.G),
                                          static_cast<int>(prob.K),
                                          static_cast<int>(prob.C),
                                          static_cast<int>(prob.Y),
                                          static_cast<int>(prob.X)};
    ck::Array<ck::index_t, 5> d_lengths = {};

    ck::Array<ck::index_t, 5> in_strides{123887616,
                                         static_cast<int>(prob.Hi * prob.Wi * prob.G * prob.C),
                                         1,
                                         static_cast<int>(prob.Wi * prob.G * prob.C),
                                         static_cast<int>(prob.G * prob.C)};
    ck::Array<ck::index_t, 5> out_strides{42467328,
                                          static_cast<int>(prob.Ho * prob.Wo * prob.G * prob.K),
                                          1,
                                          static_cast<int>(prob.Wo * prob.G * prob.K),
                                          static_cast<int>(prob.G * prob.K)};
    ck::Array<ck::index_t, 5> wei_strides{442368,
                                          static_cast<int>(prob.Y * prob.X * prob.C),
                                          1,
                                          static_cast<int>(prob.X * prob.C),
                                          static_cast<int>(prob.C)};
    ck::Array<ck::index_t, 5> d_strides = {};

    ck::Array<ck::index_t, 2> conv_filter_strides   = {2, 2};
    std::vector<ck::index_t> conv_filter_strides_   = {2, 2};
    ck::Array<ck::index_t, 2> conv_filter_dilations = {1, 1};
    std::vector<ck::index_t> conv_filter_dilations_ = {1, 1};
    ck::Array<ck::index_t, 2> input_left_pads       = {1, 1};
    std::vector<ck::index_t> input_left_pads_       = {1, 1};
    ck::Array<ck::index_t, 2> input_right_pads      = {1, 1};
    std::vector<ck::index_t> input_right_pads_      = {1, 1};

    auto get_num_elems = [](const auto& tensor_lens) {
        return std::reduce(
            tensor_lens.begin(), tensor_lens.end(), 1, std::multiplies<ck::index_t>{});
    };

    auto in_dev  = to_gpu(generate_buffer<ck::half_t>(get_num_elems(in_lengths), 0));
    auto wei_dev = to_gpu(generate_buffer<ck::half_t>(get_num_elems(wei_lengths), 1));
    auto out_dev = to_gpu(generate_buffer<ck::half_t>(get_num_elems(out_lengths), 2));

    /**constexpr ck::index_t NumATensor =
        ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();
    constexpr ck::index_t NumBTensor =
        ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();
    auto as_grid_desc_ak0_m_ak1 =
        generate_tuple([&](auto) { return arg.a_grid_desc_ak0_m_ak1_; }, ck::Number<NumATensor>{});
    auto bs_grid_desc_bk0_n_bk1 =
        generate_tuple([&](auto) { return arg.b_grid_desc_bk0_n_bk1_; },
    ck::Number<NumBTensor>{});**/

    std::cout << "entering loop" << std::endl;

    for(auto solution : prob.GetSolutions("gfx908", prologue, epilogue))
    {
        auto src = ck::host::InterpolateString(
            conv_compile_check,
            {{"include",
              "ck/tensor_operation/gpu/device/impl/"
              "copy_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"},
             {"template", solution.ToTemplateString()}});

        std::ofstream ofh("kernel.txt");
        ofh << "##########################################################\n";
        ofh << src;
        ofh << "##########################################################\n";
        ofh.close();

        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        auto name           = solution.GetTemplateParameter<std::string>("name");
        options.kernel_name = "run_" + name;
        auto k              = rtc::compile_kernel(srcs, options);

        auto block_size  = solution.GetTemplateParameter<ck::index_t>("BlockSize");
        auto m_per_block = solution.GetTemplateParameter<ck::index_t>("MPerBlock");
        auto n_per_block = solution.GetTemplateParameter<ck::index_t>("NPerBlock");
        auto num_dim     = solution.GetTemplateParameter<ck::index_t>("NumDim");
        // auto a_layout       =
        // solution.GetTemplateParameter<ck::tensor_layout::convolution::NHWGC>("ALayout");

        // decltype(arg)::foo = 1;
        // decltype(arg.a_grid_desc_ak0_m_ak1_)::foo = 1;
        // DeviceConv::AGridDesc_AK0_M_AK1::foo = 1;
        // E grid desc + block 2 etile
        const ck::index_t N = out_lengths[1];
        const ck::index_t K = out_lengths[2];

        const auto KStride         = I1;
        const ck::index_t WoStride = out_strides[num_dim + 2];

        const ck::index_t NHoWo = N * ck::accumulate_n<ck::index_t>(
                                          out_lengths.begin() + 3, num_dim, 1, std::multiplies<>());
        const auto out_gemmm_gemmn_desc = make_naive_tensor_descriptor(
            ck::make_tuple(NHoWo, K), ck::make_tuple(WoStride, KStride));
        // hard-code PadM/N = true for now: MNK Padding
        auto out = ck::tensor_operation::device::PadTensorDescriptor(
            out_gemmm_gemmn_desc,
            ck::make_tuple(m_per_block, n_per_block),
            ck::Sequence<true, true>{});
        // Grid size calculation
        const ck::index_t M0 = ck::math::integer_divide_ceil(out.GetLength(I0), m_per_block);
        const ck::index_t N0 = ck::math::integer_divide_ceil(out.GetLength(I1), n_per_block);

        auto grid_size = M0 * N0 * in_lengths[1];

        k.launch(nullptr, grid_size * block_size, block_size)(in_dev.data(),
                                                              wei_dev.data(),
                                                              out_dev.data(),
                                                              in_lengths,
                                                              in_strides,
                                                              wei_lengths,
                                                              wei_strides,
                                                              out_lengths,
                                                              out_strides,
                                                              conv_filter_strides,
                                                              conv_filter_dilations,
                                                              input_left_pads,
                                                              input_right_pads);

        Tensor<ck::half_t> in_host(in_lengths, in_strides);
        in_host.GenerateTensorValue(GeneratorTensor_1<ck::half_t>{1});
        Tensor<ck::half_t> wei_host(wei_lengths, wei_strides);
        wei_host.GenerateTensorValue(GeneratorTensor_1<ck::half_t>{1});
        Tensor<ck::half_t> out_host(out_lengths, out_strides);

        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<
            2,
            ck::half_t,
            ck::half_t,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            CDEElementOp>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in_host,
                                                  wei_host,
                                                  out_host,
                                                  conv_filter_strides_,
                                                  conv_filter_dilations_,
                                                  input_left_pads_,
                                                  input_right_pads_,
                                                  ck::tensor_operation::element_wise::PassThrough{},
                                                  ck::tensor_operation::element_wise::PassThrough{},
                                                  CDEElementOp{1.0f, 1.0f});
        std::cout << "Ref args" << std::endl;
        ref_argument.Print();

        ref_invoker.Run(ref_argument);

        bool pass = true;
        auto res  = rtc::from_gpu(out_dev);
        std::ofstream ofh2("res.txt");
        pass &= ck::utils::check_err(res, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
        ofh2 << "Check: " << pass << std::endl;
        ofh2 << res.size();
        for(int i = 0; i < res.size(); i++)
        {
            auto tmp = (res.data())[i];
            ofh2 << std::to_string(static_cast<int>(tmp)) << ", ";
        }
        ofh2.close();
        assert(pass);
        CHECK(report(solution, check(res)));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }