import enum
import os.path
import shutil
import functools
import operator
import collections
import re

def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


class EmitGemmInstance:
    def __init__(self):
        self.gemm_kernel_template =     """
        template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_K0_M0_M1_K1,
          typename BGridDesc_K0_N0_N1_K1,
          typename CGridDesc_M0_M10_M11_N0_N10_N11,
          typename Block2CTileMap,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dl_v1r3(const ${type_ab}* __restrict__ ${p_a_grid},
                            const ${type_ab}* __restrict__ ${p_b_grid},
                            ${type_c}* __restrict__ ${p_c_grid},
                            const ${A_GridDesc_K0_M_K1} ${a_grid_desc_k0_m0_m1_k1},
                            const ${BGridDesc_K0_N_K1} ${b_grid_desc_k0_n0_n1_k1},
                            const ${CGridDesc_M0_M10_M11_N0_N10_N11} ${c_grid_desc_m0_m10_m11_n0_n10_n11},
                            const Block2CTileMap ${block_2_ctile_map})
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(${type_ab});

    __shared__ ${type_ab} p_shared_block[shared_block_size];

    GridwiseGemm::Run(${p_a_grid},
                      ${p_b_grid},
                      ${p_c_grid},
                      p_shared_block,
                      ${a_grid_desc_k0_m0_m1_k1},
                      ${b_grid_desc_k0_n0_n1_k1},
                      ${c_grid_desc_m0_m10_m11_n0_n10_n11},
                      ${block_2_ctile_map},
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
}

template <index_t BlockSize,
          ${type_ab},
          ${type_acc},
          ${type_c},
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          ${A_GridDesc_K0_M_K1},
          ${BGridDesc_K0_N_K1},
          ${CGridDesc_M_N},
          ${mperblock},
          ${nperblock},
          ${k0perblock},
          ${k1value},
          ${M1PerThreadM111},
          ${N1PerThreadN111},
          ${KPerThread},
          ${M11N11ThreadClusterM110Xs},
          ${M11N11ThreadClusterN110Xs},
          ${ABlockTransferThreadSliceLengths_K0_M0_M1_K1},
          ${ABlockTransferThreadClusterLengths_K0_M0_M1_K1},
          ${ABlockTransferThreadClusterArrangeOrder},
          ${ABlockTransferSrcAccessOrder},
          ${ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1},
          ${ABlockTransferSrcVectorTensorContiguousDimOrder},
          ${ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1},
          ${BBlockTransferThreadSliceLengths_K0_N0_N1_K1},
          ${BBlockTransferThreadClusterLengths_K0_N0_N1_K1},
          ${BBlockTransferThreadClusterArrangeOrder},
          ${BBlockTransferSrcAccessOrder},
          ${BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1},
          ${BBlockTransferSrcVectorTensorContiguousDimOrder},
          ${BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1},
          ${CThreadTransferSrcDstAccessOrder},
          ${CThreadTransferSrcDstVectorDim},
          ${CThreadTransferDstScalarPerVector}>
          struct GridwiseGemmDl_km_kn_mn_v1r3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    // K1 should be Number<...>
    static constexpr auto K1 = Number<K1Value>{};

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // TODO: change this. I think it needs multi-dimensional alignment
        constexpr auto max_lds_align = K1;

        // TODO: check alignment
        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_k_m = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);

        // TODO: check alignment
        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_k_n = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);

        // TODO: check alignment
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size =
            math::integer_least_multiple(a_block_desc_k_m.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_aligned_space_size =
            math::integer_least_multiple(b_block_desc_k_n.GetElementSpaceSize(), max_lds_align);

        return 2 * (a_block_aligned_space_size + b_block_aligned_space_size) * sizeof(FloatAB);
    }
          """

def emit(self):
        values = {
            'function_name': "gemm",
            'type_a' : 'ck::half_t',
            'type_b' : 'ck::half_t',
            'type_c' : 'ck::half_t',
            'type_acc' : 'float',
            'layout_a' : 'ck::tensor_layout::gemm::RowMajor',
            'layout_b' : 'ck::tensor_layout::gemm::RowMajor',
            'layout_c' : 'ck::tensor_layout::gemm::RowMajor',
            'elementwise_op_a' : 'ck::tensor_operation::element_wise::PassThrough',
            'elementwise_op_b' : 'ck::tensor_operation::element_wise::PassThrough',
            'elementwise_op_c' : 'ck::tensor_operation::element_wise::PassThrough',
            'Gemm_spec' : 'ck::tensor_operation::device::GemmSpecialization::MNKPadding',
            'block_size' : '256',
            'mperblock' : '64',
            'nperblock' : '128',
            'kperblock' : '32',
            'k1' : '8',
            'mperxdl' : '32',
            'nperxdl' : '32',
            'mxdlperwave' : '1',
            'nxdlperwave' : '2',
            'threadclusterlength_a' : 'ck::Sequence<4,64,1>',
            'threadclusterarrange_a' : 'ck::Sequence<1,0,2>',
            'srcaccessorder_a' : 'ck::Sequence<1,0,2>',
            'srcvectordim_a' : '2',
            'srcscalarpervec_a' : '8',
            'dstscalarpervec_a' : '8',
            'add_extra_dim_a' : '1',
            'threadclusterlength_b' : 'ck::Sequence<8,32,1>',
            'threadclusterarrange_b' : 'ck::Sequence<0,2,1>',
            'srcaccessorder_b' : 'ck::Sequence<0,2,1>',
            'srcvectordim_b' : '1',
            'srcscalarpervec_b' : '4',
            'dstscalarpervec_b' : '2',
            'add_extra_dim_b' : '0',
            'dstscalarpervec_c' : '8'
        }
        template = self.gemm_template
        print(SubstituteTemplate(template, values))