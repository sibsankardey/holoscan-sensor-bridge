/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Portions Copyright (c) 2026 Analog Devices, Inc.
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

#include <cufft.h>
#include <holoscan/holoscan.hpp>
#include <vector>

namespace hololink::operators {

class AudioWaveformOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(AudioWaveformOp);

    void setup(holoscan::OperatorSpec& spec) override;

    // We add a stop() method to clean up the FFT memory when the app closes
    void stop() override;

    void compute(holoscan::InputContext& input, holoscan::OutputContext& output,
        holoscan::ExecutionContext& context) override;

private:
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<int> num_channels_;
    holoscan::Parameter<float> gain_;
    holoscan::Parameter<bool> apply_window_;
    holoscan::Parameter<float> min_freq_;
    holoscan::Parameter<float> max_freq_;
    holoscan::Parameter<int> sample_rate_;
    holoscan::Parameter<bool> verbose_;

    int samples_processed_ = 0; // Tracks elapsed time for the 1-second SNR print
    int frames_processed_ = 0; // Tracks number of 20ms frames accumulated

    // Persistent buffer for periodogram averaging
    std::vector<std::vector<double>> accumulated_power_;

    // Internal state for cuFFT
    cufftHandle fft_plan_ = 0;
    cufftReal* d_fft_in_ = nullptr;
    cufftComplex* d_fft_out_ = nullptr;
    size_t allocated_samples_ = 0;

    double* d_accumulated_power_ = nullptr;

    // Tiny device pointers for the 1Hz reduction outputs
    double* d_out_sum_ = nullptr;
    double* d_out_max_val_ = nullptr;
    int* d_out_max_idx_ = nullptr;
};

} // namespace hololink::operators
