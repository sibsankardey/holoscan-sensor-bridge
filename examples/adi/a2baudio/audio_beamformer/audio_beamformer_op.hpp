/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>

// =====================================================================
// Clockworks mic configuration - all in cms relative to reference mic
// =====================================================================
// TDM4 Slot Mapping:
// Ch 0 (mic3) = Physical Right
// Ch 1 (mic2) = Physical Left
// Ch 2 (mic1) = Physical Top
// Ch 3 (mic0) = Physical Center (Reference)
#define ARRAY_PHYSICAL_MICS 4
#define ARRAY_REF_MIC 3
#define ARRAY_MIC_X               \
    {                             \
        3.68f, -3.68f, 0.0f, 0.0f \
    }
#define ARRAY_MIC_Y                   \
    {                                 \
        -2.125f, -2.125f, 4.25f, 0.0f \
    }

// Physical Array Max Distance = 4.25 cm (~6 samples at 48kHz).
// 8 provides the perfect boundary + sub-sample interpolation padding.
#define TRACKING_MAX_LAG 8

namespace hololink::operators {

class AudioBeamformerOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(AudioBeamformerOp);

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

private:
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<int> num_channels_;
    holoscan::Parameter<int> sample_rate_;
    holoscan::Parameter<float> squelch_threshold_;
    holoscan::Parameter<float> ema_alpha_;
    holoscan::Parameter<int> hang_frames_;
    holoscan::Parameter<bool> verbose_;

    // FIR Primitives and Mic Geometry
    float* d_filtered_input_ = nullptr; // Buffer for the high-pass filtered audio
    bool fir_initialized_ = false; // Flag to copy constants only once
    float* d_fir_coeffs_ = nullptr;

    // NEW: Persistent IIR Filter State
    float* d_iir_x_ = nullptr;
    float* d_iir_y_hp_ = nullptr;
    float* d_iir_y_lp_ = nullptr;
    bool first_run_ = true;

    float* d_mic_magnitudes_ = nullptr; // Array of size [num_channels]
    float* d_mic_max_partial_ = nullptr; // For magnitude reduction

    float* d_correlation_results_ = nullptr; // [num_mics * (2 * max_lag + 1)]
    int* d_steering_lags_ = nullptr; // [num_channels]
    float* d_fractional_lags_ = nullptr;
    float* d_peak_vals_ = nullptr;

    // Geometry logic
    float last_azimuth_ = 0.0f;
    float last_elevation_ = 0.0f;
    float last_dbfs_ = -100.0f;

    int samples_processed_ = 0;
    float* d_partial_sums_ = nullptr; // Buffer for parallel reduction
    double total_squared_sum_ = 0.0;
    int snr_samples_count_ = 0;
    int squelch_hold_frames_ = 0;
};

} // namespace hololink::operators
