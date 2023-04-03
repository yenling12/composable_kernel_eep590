
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
// #include <half.hpp>
#include <random>
#include <rocrand/rocrand.h>
#include "logging.h"
#include "include/ck/utility/print.hpp"
#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"
#include "include/ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/tensor_operation/gpu/element/element_wise_operation.hpp"


#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"







using gemm_2_hhh_TTT_256_64_128_32_8_2_32_32_1_2_PT = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<

    ck::tensor_layout::gemm::RowMajor,
    ck::tensor_layout::gemm::RowMajor,
    ck::tensor_layout::gemm::RowMajor,
    ck::half_t,
    ck::half_t,
    ck::half_t,
    float,
    ck::half_t,


    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,

    ck::tensor_operation::device::GemmSpecialization::MNKPadding,

    1,


   256, // block_size
   64, // m_per_block
   128, // n_per_block
   32, // k_per_block
   8, // ak1
   2, // bk1
   32, // m_per_xdl
   32, // n_per_xdl
   1, // m_xdl_per_wave
   2, // n_xdl_per_wave


    ck::Sequence<4,64,1>, // thread_cluster_length
    ck::Sequence<1,0,2>, // thread_cluster_arrange_order
    ck::Sequence<1,0,2>, // src_access_order
    2, // src_vector_dim
    8, // src_scalar_per_vector
    8, // dst_scalar_per_vector

    1, // add_extra_dim


    ck::Sequence<8,32,1>, // thread_cluster_length
    ck::Sequence<0,2,1>, // thread_cluster_arrange_order
    ck::Sequence<0,2,1>, // src_access_order
    1, // src_vector_dim
    4, // src_scalar_per_vector
    2, // dst_scalar_per_vector

    0, // add_extra_dim
    1, // m_xdl_per_wave
    1, // n_xdl_per_wave
    ck::Sequence<1,32,1,8>, // m_n_block_wave_per_xdl
    8 // scalar_per_vector


    >;
using fe7d3cbb34ba481ca532c1fecec7ec7cc5fbb35d0 = gemm_2_hhh_TTT_256_64_128_32_8_2_32_32_1_2_PT;


void gemm_rrr_3(
    void * in_ptr,
    void * weight_ptr,
    void * out_ptr,




    int64_t* a_dim0,

    int64_t* a_dim1,


    int64_t* b_dim0,

    int64_t* b_dim1,


    int64_t* c_dim0,

    int64_t* c_dim1,


    hipStream_t stream
    ) {

 ck::index_t M = (*a_dim0);

 ck::index_t N = (*b_dim1);

 ck::index_t K = (*a_dim1);
  int64_t offset_a = 0;
  int64_t offset_b = 0;
  int64_t offset_c = 0;

  ck::index_t stride_a = *a_dim1;
  ck::index_t stride_b = *b_dim1;
  ck::index_t stride_c = *c_dim1;

  if (M == 256 && N == 32 && K == 128) {

    auto op =  fe7d3cbb34ba481ca532c1fecec7ec7cc5fbb35d0{};
    auto invoker  = op.MakeInvoker();
    auto argument = op.MakeArgument(

                                    static_cast<ck::half_t *>(in_ptr) + offset_a,
                                    static_cast<ck::half_t *>(weight_ptr) + offset_b,




                                    static_cast<ck::half_t *>(out_ptr) + offset_c,

                                    M,
                                    N,
                                    K,
                                    stride_a,
                                    stride_b,



                                    stride_c,
                                    ck::tensor_operation::element_wise::PassThrough{},
                                    ck::tensor_operation::element_wise::PassThrough{},

                                    ck::tensor_operation::element_wise::PassThrough{}

    );
    if(!op.IsSupportedArgument(argument)) {
      LOG(FATAL) << "wrong! " << op.GetTypeString() << " with the specified compilation parameters does not support this Gemm problem.";
    }

    invoker.Run(argument, StreamConfig{stream, false});
    return;
  }

  LOG(FATAL) << "Unsupported workload for this gemm specialization.";
}