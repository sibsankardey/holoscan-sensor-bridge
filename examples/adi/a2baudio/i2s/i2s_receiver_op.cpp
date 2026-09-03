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

#include "i2s_receiver_op.hpp"

#include <chrono>
#include <fmt/format.h>
#include <gxf/core/entity.hpp>
#include <gxf/std/tensor.hpp>
#include <hololink/core/logging_internal.hpp>
#include <thread>

namespace hololink::operators {

void I2sReceiverOp::setup(holoscan::OperatorSpec& spec)
{
    // 1. Call the base class setup
    hololink::operators::LinuxReceiverOp::setup(spec);

    // 2. Register ALL of our custom parameters here
    spec.param(i2s_channel_, "i2s_channel", "I2S Channel", "Pointer to the DataChannel");
    spec.param(i2s_address_, "i2s_address", "I2S Address", "Memory address for the I2S IP block");
    spec.param(chunk_size_, "chunk_size", "Chunk Size", "I2S chunk size (e.g., 256)", 256u);
    spec.param(i2s_cfg_value_, "i2s_cfg_value", "I2S Config Value", "Configuration bitmask for I2S_CFG", 0x00042020u);
}
void I2sReceiverOp::start()
{
    hololink::operators::LinuxReceiverOp::start();

    auto channel = i2s_channel_.get();
    auto hololink = channel->hololink();
    try {
        i2s_ = hololink->get_i2s(channel->enumeration_metadata());
    } catch (...) {
        HSB_LOG_WARN("I2S get_i2s() not available, operator will use direct register access fallback.");
        i2s_ = nullptr;
    }

    if (!i2s_) {
        uint32_t cfg_val = i2s_cfg_value_.get();
        hololink->write_uint32(i2s_address_.get() + 0x04, chunk_size_.get());
        hololink->write_uint32(i2s_address_.get() + 0x08, cfg_val);
    }
}

void I2sReceiverOp::start_hardware()
{
    if (hardware_started_)
        return;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    i2s_channel_.get()->hololink()->write_uint32(i2s_address_.get() + 0x00, 1);
    hardware_started_ = true;
}

void I2sReceiverOp::stop_hardware()
{
    if (!hardware_started_)
        return;

    i2s_channel_.get()->hololink()->write_uint32(i2s_address_.get() + 0x00, 0);
    hardware_started_ = false;
}

void I2sReceiverOp::compute(holoscan::InputContext& input, holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    auto maybe_cuda_stream = context.allocate_cuda_stream();
    if (!maybe_cuda_stream) {
        throw std::runtime_error("Failed to allocate CUDA stream");
    }
    const auto cuda_stream = maybe_cuda_stream.value();

    const double timeout_ms = 5000.0;

    auto [frame_device_ptr, frame_metadata] = get_next_frame(timeout_ms, cuda_stream);
    if (!frame_metadata) {
        HSB_LOG_WARN("I2S Receiver Timeout! No data received from Hololink in {} ms.", timeout_ms);
        timeout(input, op_output, context);
        return;
    }

    frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    frame_count_++;

    int tensor_size = static_cast<int>(frame_size_.get());
    if (trim_.get()) {
        auto bytes_written_opt = frame_metadata->get<int64_t>("bytes_written");
        if (bytes_written_opt && *bytes_written_opt > 0 && static_cast<size_t>(*bytes_written_opt) <= static_cast<size_t>(tensor_size)) {
            tensor_size = static_cast<int>(*bytes_written_opt);
        }
    }
    HSB_LOG_DEBUG("SUCCESS: I2S Frame {} received! Payload size: {} bytes", frame_count_, tensor_size);

    nvidia::gxf::Expected<nvidia::gxf::Entity> out_message = nvidia::gxf::Entity::New(context.context());
    if (!out_message)
        throw std::runtime_error("Failed to create GXF entity");

    nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Tensor>> gxf_tensor = out_message.value().add<nvidia::gxf::Tensor>("");
    if (!gxf_tensor)
        throw std::runtime_error("Failed to add GXF tensor");

    const nvidia::gxf::Shape shape { tensor_size };
    const nvidia::gxf::PrimitiveType element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    const uint64_t element_size = nvidia::gxf::PrimitiveTypeSize(element_type);

    if (!gxf_tensor.value()->wrapMemory(shape, element_type, element_size,
            nvidia::gxf::ComputeTrivialStrides(shape, element_size),
            nvidia::gxf::MemoryStorageType::kDevice, reinterpret_cast<void*>(frame_device_ptr),
            [](void*) { return nvidia::gxf::Success; })) {
        throw std::runtime_error("Failed to wrap memory");
    }

    auto const& meta = metadata();
    for (const auto& [key, value] : *frame_metadata) {
        if (std::holds_alternative<int64_t>(value))
            meta->set(key, std::get<int64_t>(value));
        else if (std::holds_alternative<std::string>(value))
            meta->set(key, std::get<std::string>(value));
        else if (std::holds_alternative<std::vector<uint8_t>>(value))
            meta->set(key, std::get<std::vector<uint8_t>>(value));
    }

    op_output.set_cuda_stream(cuda_stream, "output");
    op_output.emit(out_message.value(), "output");
}

} // namespace hololink::operators
