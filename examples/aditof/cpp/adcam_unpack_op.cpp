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

#include "adcam_unpack_op.hpp"

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <holoscan/holoscan.hpp>

#include <gxf/core/entity.hpp>
#include <gxf/std/tensor.hpp>

#include <cuda_runtime.h>
#include <fstream>
#include <vector>

//------------------------------------------------------------------------------
// CUDA error checking helper
//------------------------------------------------------------------------------
#define CudaCheckRuntime(FUNC)                                 \
    {                                                          \
        cudaError_t err = FUNC;                                \
        if (err != cudaSuccess) {                              \
            throw std::runtime_error(cudaGetErrorString(err)); \
        }                                                      \
    }

//------------------------------------------------------------------------------
// Save raw packed frame (device → host → file)
//------------------------------------------------------------------------------
bool should_save_raw_packed(const std::string& filename,
    uint8_t* device_ptr,
    size_t bytes,
    cudaStream_t stream)
{
    std::vector<uint8_t> host_buffer(bytes);

    cudaMemcpyAsync(host_buffer.data(),
        device_ptr,
        bytes,
        cudaMemcpyDeviceToHost,
        stream);
    cudaStreamSynchronize(stream);

    if (host_buffer[0xF0] != 0x0) {
        return true;
    }
    // std::ofstream ofs(filename, std::ios::binary);
    // ofs.write(reinterpret_cast<char*>(host_buffer.data()), bytes);
    HOLOSCAN_LOG_INFO("Received invalid data");
    return false;
}

//------------------------------------------------------------------------------
// Save raw packed frame (device → host → file)
//------------------------------------------------------------------------------
void save_raw_packed(const std::string& filename,
    uint8_t* device_ptr,
    size_t bytes,
    cudaStream_t stream)
{
    std::vector<uint8_t> host_buffer(bytes);

    cudaMemcpyAsync(host_buffer.data(),
        device_ptr,
        bytes,
        cudaMemcpyDeviceToHost,
        stream);
    cudaStreamSynchronize(stream);

    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<char*>(host_buffer.data()), bytes);
}

//==============================================================================
// ADTFUnpackOp Implementation
//==============================================================================
namespace hololink::operators {

//------------------------------------------------------------------------------
// Setup: declare inputs, outputs, parameters
//------------------------------------------------------------------------------
void ADTFUnpackOp::setup(holoscan::OperatorSpec& spec)
{
    HOLOSCAN_LOG_DEBUG("ADTFUnpackOp setup start");
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(width_, "width");
    spec.param(height_, "height");
    spec.param(num_planes_, "num_planes");

    spec.param(z_table_, "z_table", "Radial to Z table",
        "Per-pixel scale converting radial depth to Cartesian Z; "
        "empty reports the radial depth as received from the ADSD3500",
        std::vector<float>());

    spec.param(ab_log_scale_, "ab_log_scale", "Log scale active brightness",
        "Display active brightness on a logarithmic scale instead of linear",
        true);

    spec.param(depth_min_mm_, "depth_min_mm", "Depth colormap minimum",
        "Depth in mm at the bottom of the Jet colormap; closer pixels clamp "
        "to it",
        kDepthMinMmDefault);

    spec.param(depth_max_mm_, "depth_max_mm", "Depth colormap maximum",
        "Depth in mm at the top of the Jet colormap; farther pixels and "
        "invalid pixels are drawn black",
        kDepthMaxMmDefault);

    spec.param(fps_interval_sec_, "profile_avg_fps", "Profiling of FPS over seconds", "Average FPS processed", 0);

    spec.param(allocator_, "allocator", "Allocator",
        "Device allocator for output tensors");

    // Python equivalent: in_message.get("")
    spec.param(in_tensor_name_, "in_tensor_name", "",
        "Name of the input tensor ('' = unnamed)", std::string(""));

    spec.param(out_tensor_name_, "out_tensor_name", "output",
        "Name of the output port", std::string("output"));

    cuda_stream_handler_.define_params(spec);

    // For mode 0 &1, pixel size is 4, otherwise 5
    //

    HOLOSCAN_LOG_DEBUG("ADTFUnpackOp setup complete");
}

//------------------------------------------------------------------------------
// Start: compute frame size
//------------------------------------------------------------------------------
void ADTFUnpackOp::start()
{
    frame_size_ = width_.get() * height_.get();
    pixel_size_ = num_planes_ == 2 ? 4 : 5;

    // For profiling
    fps_window_start_ = Clock::now();
    frames_received_ = 0;
    frames_processed_ = 0;
    total_processing_us_ = 0;

    const std::vector<float>& z_table = z_table_.get();
    if (!z_table.empty()) {
        if (static_cast<int>(z_table.size()) != frame_size_) {
            throw std::runtime_error(fmt::format(
                "z_table has {} entries, expected {} ({}x{})",
                z_table.size(), frame_size_, width_.get(), height_.get()));
        }
        const size_t bytes = z_table.size() * sizeof(float);
        CudaCheckRuntime(cudaMalloc(&z_table_device_, bytes));
        CudaCheckRuntime(cudaMemcpy(z_table_device_, z_table.data(), bytes,
            cudaMemcpyHostToDevice));
        HOLOSCAN_LOG_INFO("ADTFUnpackOp: reporting Cartesian Z depth");
    } else {
        HOLOSCAN_LOG_INFO("ADTFUnpackOp: reporting radial depth");
    }

    const float depth_min_mm = depth_min_mm_.get();
    const float depth_max_mm = depth_max_mm_.get();
    const float tick_step_mm = depth_legend_tick_step_mm(depth_min_mm, depth_max_mm);
    const float first_tick_mm = depth_legend_first_tick_mm(depth_min_mm, depth_max_mm);
    const int ticks = depth_legend_ticks(depth_min_mm, depth_max_mm);

    std::vector<float> legend_coords(ticks * 3);
    for (int i = 0; i < ticks; ++i) {
        const float value_mm = first_tick_mm + i * tick_step_mm;
        legend_coords[i * 3 + 0] = kDepthLegendTextX;
        legend_coords[i * 3 + 1] = kDepthLegendBarY
            + kDepthLegendBarHeight
                * (1.0f - (value_mm - depth_min_mm) / (depth_max_mm - depth_min_mm))
            - kDepthLegendTextSize * 0.5f;
        legend_coords[i * 3 + 2] = kDepthLegendTextSize;
    }
    legend_coords_size_ = legend_coords.size() * sizeof(float);
    CudaCheckRuntime(cudaMalloc(&legend_coords_d_, legend_coords_size_));
    CudaCheckRuntime(cudaMemcpy(legend_coords_d_, legend_coords.data(), legend_coords_size_,
        cudaMemcpyHostToDevice));

    HOLOSCAN_LOG_DEBUG("ADTFUnpackOp start complete");
}

//------------------------------------------------------------------------------
void ADTFUnpackOp::stop()
{
    if (z_table_device_) {
        cudaFree(z_table_device_);
        z_table_device_ = nullptr;
    }
    if (legend_coords_d_) {
        cudaFree(legend_coords_d_);
        legend_coords_d_ = nullptr;
    }
    HOLOSCAN_LOG_DEBUG("ADTFUnpackOp stop complete");
}

//------------------------------------------------------------------------------
// Main compute function
//------------------------------------------------------------------------------
void ADTFUnpackOp::compute(holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{

    auto frame_start = Clock::now();

    ++frames_received_;
    //--------------------------------------------------------------------------
    // 1. Receive input entity
    //--------------------------------------------------------------------------
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input entity");
    }
    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    //--------------------------------------------------------------------------
    // 2. Extract CUDA stream from message
    //--------------------------------------------------------------------------
    gxf_result_t stream_handler_result = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "Failed to get CUDA stream: {}", GxfResultStr(stream_handler_result)));
    }
    cudaStream_t stream = cuda_stream_handler_.get_cuda_stream(context.context());

    //--------------------------------------------------------------------------
    // 3. Get input tensor (Python: msg = in_message.get(""))
    //--------------------------------------------------------------------------
    auto maybe_tensor = entity.get<nvidia::gxf::Tensor>(in_tensor_name_.get().c_str());
    if (!maybe_tensor) {
        throw std::runtime_error(fmt::format(
            "Input tensor '{}' not found", in_tensor_name_.get()));
    }
    auto input_tensor = maybe_tensor.value();

    //--------------------------------------------------------------------------
    // 4. Validate storage type
    //--------------------------------------------------------------------------
    if (input_tensor->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
        HOLOSCAN_LOG_WARN("Input tensor is in host memory — slower performance.");
    } else if (input_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        throw std::runtime_error("Unsupported tensor storage type");
    }

    const int width = width_.get();
    const int height = height_.get();
    const int size = width * height;

    // Expect 5 bytes per pixel (ADI ToF packed format)
    const size_t expected_bytes = static_cast<size_t>(size * pixel_size_);
    if (input_tensor->size() < expected_bytes) {
        throw std::runtime_error(fmt::format(
            "Input tensor too small: {} bytes, expected {}", input_tensor->size(), expected_bytes));
    }

    HOLOSCAN_LOG_DEBUG("Input tensor: {} bytes, {} elements",
        input_tensor->size(), input_tensor->element_count());

    //--------------------------------------------------------------------------
    // 5. Get allocator handle
    //--------------------------------------------------------------------------
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());
    if (!allocator) {
        throw std::runtime_error("Failed to get allocator handle");
    }

    //--------------------------------------------------------------------------
    // 6. Create output entity and RGB output tensors
    //--------------------------------------------------------------------------
    auto out_message = nvidia::gxf::Entity::New(context.context()).value();

    auto depth_tensor = out_message.add<nvidia::gxf::Tensor>("Depth").value();
    auto ab_tensor = out_message.add<nvidia::gxf::Tensor>("ActiveBrightness").value();
    nvidia::gxf::Handle<nvidia::gxf::Tensor> conf_tensor;
    uint8_t* conf_rgb_ptr = nullptr;
    if (num_planes_ != 2) {
        conf_tensor = out_message.add<nvidia::gxf::Tensor>("Conf").value();
        conf_tensor->reshape<uint8_t>({ height, width, 3 },
            nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
        conf_rgb_ptr = conf_tensor->data<uint8_t>().value();
    }

    depth_tensor->reshape<uint8_t>({ height, width, 3 },
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
    ab_tensor->reshape<uint8_t>({ height, width, 3 },
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

    auto legend_tensor = out_message.add<nvidia::gxf::Tensor>("DepthLegend").value();
    legend_tensor->reshape<float>(
        { depth_legend_ticks(depth_min_mm_.get(), depth_max_mm_.get()), 3 },
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

    uint8_t* depth_rgb_ptr = depth_tensor->data<uint8_t>().value();
    uint8_t* ab_rgb_ptr = ab_tensor->data<uint8_t>().value();

    //--------------------------------------------------------------------------
    // 7. Interpret input as uint16 (Python: cp.asarray(msg))
    //--------------------------------------------------------------------------
    uint16_t* raw_u16 = input_tensor->data<uint16_t>().value();

    // raw_u16 += 128; //remove metadata frame

    //--------------------------------------------------------------------------
    // 8. Allocate internal unpack buffers (uint16)
    //--------------------------------------------------------------------------

    auto scratch_entity = nvidia::gxf::Entity::New(context.context()).value();

    auto depthraw_tensor = scratch_entity.add<nvidia::gxf::Tensor>("depthraw").value();
    auto abraw_tensor = scratch_entity.add<nvidia::gxf::Tensor>("abraw").value();
    nvidia::gxf::Handle<nvidia::gxf::Tensor> confraw_tensor; //  = scratch_entity.add<nvidia::gxf::Tensor>("confraw").value();
    uint16_t* conf = nullptr;
    if (num_planes_ != 2) {
        confraw_tensor = scratch_entity.add<nvidia::gxf::Tensor>("abraw").value();
        confraw_tensor->reshape<uint16_t>({ height, width },
            nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
        conf = confraw_tensor->data<uint16_t>().value();
    }

    // nvidia::gxf::Tensor depthraw_tensor, confraw_tensor, abraw_tensor;

    depthraw_tensor->reshape<uint16_t>({ height, width },
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
    abraw_tensor->reshape<uint16_t>({ height, width },
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

    uint16_t* depth = depthraw_tensor->data<uint16_t>().value();
    uint16_t* ab = abraw_tensor->data<uint16_t>().value();

    //--------------------------------------------------------------------------
    // 9. Convert 16-bit → 8-bit (Python: (cp_frame >> 8).astype(uint8))
    //--------------------------------------------------------------------------
    // nvidia::gxf::Tensor frame_u8_tensor;
    auto frame_u8_tensor = scratch_entity.add<nvidia::gxf::Tensor>("rawraw").value();
    frame_u8_tensor->reshape<uint8_t>({ height, width * pixel_size_ },
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

    uint8_t* raw = frame_u8_tensor->data<uint8_t>().value();

    shift_and_cast_kernel(raw_u16, raw, size * pixel_size_, stream);

    // Save raw packed frame once
    static bool saved_once = false, valid_data = false;

    // valid_data = should_save_raw_packed("packed_frame.bin", raw, expected_bytes, stream);
    if (!saved_once) { // && (valid_data == true)) {
        save_raw_packed("packed_frame.bin", raw, expected_bytes, stream);
        saved_once = true;
    }

    //--------------------------------------------------------------------------
    // 10. Unpack 5-byte/pixel → depth/conf/ab (uint16) or
    // Unpack 4-byte/pixel → depth/ab (uint16)
    //--------------------------------------------------------------------------
    if (num_planes_ == 2) {
        unpack_kernel_launch(raw, depth, nullptr, ab, width, height, stream);
    } else {
        unpack_kernel_launch(raw, depth, conf, ab, width, height, stream);
    }

    //--------------------------------------------------------------------------
    // 11. Convert to RGB (Jet + grayscale)
    //--------------------------------------------------------------------------
    const uint16_t* depth_out = depth;
    if (z_table_device_) {
        auto depthz_tensor = scratch_entity.add<nvidia::gxf::Tensor>("depthz").value();
        depthz_tensor->reshape<uint16_t>({ height, width },
            nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
        uint16_t* depth_z = depthz_tensor->data<uint16_t>().value();
        radial_to_z_kernel_launch(depth, z_table_device_, depth_z, size, stream);
        depth_out = depth_z;
    }

    jet_kernel_launch(depth_out, depth_rgb_ptr, size, stream,
        depth_min_mm_.get(), depth_max_mm_.get());
    depth_legend_kernel_launch(depth_rgb_ptr, width, height, stream,
        depth_min_mm_.get(), depth_max_mm_.get());
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("[ADTFUnpackOp] jet_kernel launch error: {}",
                cudaGetErrorString(err));
        }
    }
    grayscale_kernel_launch(conf, conf_rgb_ptr, size, stream,
        255.0f, false); // conf: 8-bit range 0-255
    grayscale_kernel_launch(ab, ab_rgb_ptr, size, stream,
        4096.0f, ab_log_scale_.get()); // AB: 12-bit range 0-4096
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            HOLOSCAN_LOG_ERROR(
                "[ADTFUnpackOp] grayscale_kernel launch error: {}",
                cudaGetErrorString(err));
        }
    }

    //--------------------------------------------------------------------------
    // 12. Position the legend tick labels
    //--------------------------------------------------------------------------
    CudaCheckRuntime(cudaMemcpy(legend_tensor->data<float>().value(),
        legend_coords_d_, legend_coords_size_,
        cudaMemcpyDeviceToDevice));

    //--------------------------------------------------------------------------
    // 13. Emit output entity
    //--------------------------------------------------------------------------
    op_output.emit(out_message);

    //--------------------------------------------------------------------------
    // 14. Update profiling info if needed/enabled
    //--------------------------------------------------------------------------
    if (fps_interval_sec_) {
        ++frames_processed_;

        auto frame_end = Clock::now();

        total_processing_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
            frame_end - frame_start)
                                    .count();

        const double elapsed_sec = std::chrono::duration<double>(
            frame_end - fps_window_start_)
                                       .count();

        if (elapsed_sec >= fps_interval_sec_) {

            uint64_t rx_count = frames_received_.load();

            uint64_t processed_count = frames_processed_.load();

            double received_fps = static_cast<double>(rx_count) / elapsed_sec;

            double processed_fps = static_cast<double>(processed_count) / elapsed_sec;

            double avg_processing_us = processed_count ? static_cast<double>(total_processing_us_) / processed_count : 0.0;

            HOLOSCAN_LOG_INFO(
                "\n"
                "=====================================================\n"
                " ADCAM TOF FPS REPORT\n"
                "=====================================================\n"
                " Window                 : {} sec\n"
                " Frames Received        : {} \n"
                " Frames Processed       : {} \n"
                " Received FPS           : {} \n"
                " Processed FPS          : {} \n"
                " Avg Processing Time    : {} us\n"
                "=====================================================",
                elapsed_sec,
                rx_count,
                processed_count,
                received_fps,
                processed_fps,
                avg_processing_us);

            frames_received_ = 0;
            frames_processed_ = 0;
            total_processing_us_ = 0;
            fps_window_start_ = frame_end;
        }
    }
}

} // namespace hololink::operators
