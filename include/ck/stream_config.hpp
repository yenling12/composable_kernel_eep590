// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

struct StreamConfig
{
    hipStream_t stream_id_ = nullptr;
    int time_kernel_      = 0;
    int log_level_         = 0;
    int cold_niters_       = 50;
    int nrepeat_           = 200;
};
