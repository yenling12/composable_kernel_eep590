// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "device_gemm_xdl_universal_f8_f8_f16_mk_nk_mn_mem_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_f8_f8_f16_mk_nk_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_gemm_xdl_universal_f8_f8_f16_mk_nk_mn_instances<GemmMNKPadding, BlockGemmPipelineScheduler::Interwave>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
