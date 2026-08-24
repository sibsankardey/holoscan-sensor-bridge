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
#include "audio_beamformer_op.hpp"
#include <cuda_runtime.h>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

void AudioBeamformerOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(allocator_, "allocator", "Allocator", "Output allocator");
    spec.param(num_channels_, "num_channels", "Channels", "Input mic count", 4);
    spec.param(sample_rate_, "sample_rate", "Sample Rate", "Hz", 48000);
    spec.param(squelch_threshold_, "squelch_threshold", "Squelch", "dBFS", -50.0f);
    spec.param(ema_alpha_, "ema_alpha", "EMA Alpha", "Smoothing", 0.3f);
    spec.param(hang_frames_, "hang_frames", "Hang Frames", "Hold duration", 10);
    spec.param(verbose_, "verbose", "Verbose Logging", "Enable console printouts", false);
}

void AudioBeamformerOp::start()
{
    int mics = num_channels_.get();
    cudaMalloc(&d_fir_coeffs_, 41 * sizeof(float));

    // 41-Tap Bandpass FIR (80Hz-3000Hz) @ 48kHz
    float h_fir[41] = {
        -0.001168f, -0.001556f, -0.001476f, 0.000000f, 0.003079f, 0.007469f, 0.012061f, 0.015187f,
        0.014463f, 0.007604f, -0.005517f, -0.023247f, -0.042316f, -0.058145f, -0.066128f, -0.062035f,
        -0.042544f, -0.006900f, 0.040994f, 0.095535f, 0.117117f, 0.095535f, 0.040994f, -0.006900f,
        -0.042544f, -0.062035f, -0.066128f, -0.058145f, -0.042316f, -0.023247f, -0.005517f, 0.007604f,
        0.014463f, 0.015187f, 0.012061f, 0.007469f, 0.003079f, 0.000000f, -0.001476f, -0.001556f, -0.001168f
    };
    cudaMemcpy(d_fir_coeffs_, h_fir, 41 * sizeof(float), cudaMemcpyHostToDevice);
}

void AudioBeamformerOp::stop()
{
    // 1. FIR Primitives
    if (d_fir_coeffs_) {
        cudaFree(d_fir_coeffs_);
        d_fir_coeffs_ = nullptr;
    }

    // 2. SNR and TDOA Buffers
    if (d_partial_sums_) {
        cudaFree(d_partial_sums_);
        d_partial_sums_ = nullptr;
    }
    if (d_correlation_results_) {
        cudaFree(d_correlation_results_);
        d_correlation_results_ = nullptr;
    }

    // 3. Steering Buffer
    if (d_steering_lags_) {
        cudaFree(d_steering_lags_);
        d_steering_lags_ = nullptr;
    }

    HSB_LOG_INFO("AudioBeamformerOp: Resources released and CUDA memory freed.");
}

} // namespace hololink::operators
