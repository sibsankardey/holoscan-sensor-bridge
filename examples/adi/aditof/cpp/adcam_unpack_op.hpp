/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hololink/core/csi_controller.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <cuda.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <vector>

void shift_and_cast_kernel(
    const uint16_t* in,
    uint8_t* out,
    int count,
    cudaStream_t stream);

void radial_to_z_kernel_launch(
    const uint16_t* radial,
    const float* z_table,
    uint16_t* z,
    int size,
    cudaStream_t stream);

void grayscale_kernel_launch(
    const uint16_t* input,
    uint8_t* rgb,
    int size,
    cudaStream_t stream,
    float max_val,
    bool log_scale);

// Default extent of the Jet colormap, overridable with the operator's
// depth_min_mm / depth_max_mm parameters; anything outside it is drawn black.
// The module cannot resolve closer than ~0.4 m.
constexpr float kDepthMinMmDefault = 400.0f;
constexpr float kDepthMaxMmDefault = 4000.0f;

// Colorbar legend layout, as fractions of the depth image / of the Holoviz view
// showing it. The bar is drawn into the depth image by the CUDA kernel, the tick
// labels are a Holoviz text layer, so both must agree on these.
constexpr float kDepthLegendBarX = 0.0f;
constexpr float kDepthLegendBarWidth = 0.045f;
constexpr float kDepthLegendBarY = 0.06f;
constexpr float kDepthLegendBarHeight = 0.88f;
constexpr float kDepthLegendTextX = 0.055f;
constexpr float kDepthLegendTextSize = 0.03f;
constexpr float kDepthLegendTickStepMm = 500.0f;

// Labels would overlap beyond this, so the tick step adapts to the span.
constexpr int kDepthLegendMaxTicks = 8;

inline float depth_legend_tick_step_mm(float depth_min_mm, float depth_max_mm)
{
    const float span = depth_max_mm - depth_min_mm;
    float step = kDepthLegendTickStepMm;
    while ((int)(span / step) > kDepthLegendMaxTicks) {
        step *= 2.0f;
    }
    while ((step > 1.0f) && (span < step)) {
        step *= 0.5f;
    }
    return step;
}

// First tick is the lowest step multiple at or above the bottom of the bar.
inline float depth_legend_first_tick_mm(float depth_min_mm, float depth_max_mm)
{
    const float step = depth_legend_tick_step_mm(depth_min_mm, depth_max_mm);
    return ceilf(depth_min_mm / step) * step;
}

inline int depth_legend_ticks(float depth_min_mm, float depth_max_mm)
{
    const float step = depth_legend_tick_step_mm(depth_min_mm, depth_max_mm);
    const float first = depth_legend_first_tick_mm(depth_min_mm, depth_max_mm);
    return (int)((depth_max_mm - first) / step) + 1;
}

void jet_kernel_launch(
    const uint16_t* depth,
    uint8_t* rgb,
    int size,
    cudaStream_t stream,
    float depth_min_mm,
    float depth_max_mm);

void depth_legend_kernel_launch(
    uint8_t* rgb,
    int width,
    int height,
    cudaStream_t stream,
    float depth_min_mm,
    float depth_max_mm);

void unpack_kernel_launch(
    const uint8_t* raw,
    uint16_t* depth,
    uint16_t* conf,
    uint16_t* ab,
    int width,
    int height,
    cudaStream_t stream);
namespace hololink::operators {

/**
 * Operator that enables the HSB packetizer to output 10- and 12-bit packed pixel formats,
 * then converts the output to 16 bits per pixel.
 *   - 10-bit format is packed 3 pixels per 4 bytes as {2'b0, p3[9:0], p2[9:0], p1[9:0]}
 *   - 12-bit format is packed 2 pixels per 3 bytes as {p2[11:0], p1[11:0]}
 */
class ADTFUnpackOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ADTFUnpackOp);
    ADTFUnpackOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;

    void compute(
        holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

private:
    holoscan::Parameter<int> width_;
    holoscan::Parameter<int> height_;
    holoscan::Parameter<int> num_planes_;

    // Per-pixel radial->Cartesian Z scale; empty leaves the depth plane radial.
    holoscan::Parameter<std::vector<float>> z_table_;

    holoscan::Parameter<bool> ab_log_scale_;

    holoscan::Parameter<float> depth_min_mm_;
    holoscan::Parameter<float> depth_max_mm_;

    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;

    int frame_size_;
    int pixel_size_;
    float* z_table_device_ = nullptr;
    float* legend_coords_d_ = nullptr;
    int legend_coords_size_ = 0;

    holoscan::Parameter<int> cuda_device_ordinal_;
    std::shared_ptr<holoscan::Tensor> depth_tensor_;
    std::shared_ptr<holoscan::Tensor> conf_tensor_;
    std::shared_ptr<holoscan::Tensor> ab_tensor_;

    std::shared_ptr<holoscan::Tensor> depth_rgb_;
    std::shared_ptr<holoscan::Tensor> conf_rgb_;
    std::shared_ptr<holoscan::Tensor> ab_rgb_;

    holoscan::Parameter<std::string> in_tensor_name_;
    holoscan::Parameter<std::string> out_tensor_name_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    holoscan::CudaStreamHandler cuda_stream_handler_;
    // FOR profiling
    using Clock = std::chrono::steady_clock;

    Clock::time_point fps_window_start_;

    std::atomic<uint64_t> frames_received_ { 0 };
    std::atomic<uint64_t> frames_processed_ { 0 };

    uint64_t total_processing_us_ = 0;
    holoscan::Parameter<int> fps_interval_sec_;
    // std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;
};

} // namespace hololink::operators
