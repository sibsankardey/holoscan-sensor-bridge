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
#include "audio_file_writer_op.hpp"
#include <cuda_runtime.h>
#include <gxf/std/tensor.hpp>
#include <hololink/core/logging_internal.hpp>
#include <math.h>
#include <stdexcept>

namespace hololink::operators {

template <typename T>
void write_bin(std::ofstream& out, const T& value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void AudioFileWriterOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.param(file_path_, "file_path", "File Path", "Path to save the .wav file", std::string("tracked_voice.wav"));
    spec.param(sample_rate_, "sample_rate", "Sample Rate", "Audio sample rate", 48000);
    spec.param(record_start_, "record_start", "Record Start", "Start time in seconds", 0.0f);
    spec.param(record_stop_, "record_stop", "Record Stop", "Stop time in seconds", 0.0f);
}

void AudioFileWriterOp::start()
{
    file_.open(file_path_.get(), std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + file_path_.get());
    }

    for (int i = 0; i < 44; ++i)
        file_.put(0);

    samples_seen_ = 0;
    samples_written_ = 0;
    dc_prev_x_ = 0.0f;
    dc_prev_y_ = 0.0f;
    recording_started_ = false;
    recording_finished_ = false;

    float start_sec = record_start_.get();
    float stop_sec = record_stop_.get();

    start_sample_ = static_cast<uint32_t>(start_sec * sample_rate_.get());
    stop_sample_ = (stop_sec > 0.0f && stop_sec > start_sec) ? static_cast<uint32_t>(stop_sec * sample_rate_.get()) : 0xFFFFFFFF;
}

void AudioFileWriterOp::update_header()
{
    if (file_.is_open()) {
        std::streampos current_pos = file_.tellp(); // Save current position at the end of the file

        // CLEAR any stream error flags so seekp is guaranteed to work
        file_.clear();
        file_.seekp(0, std::ios::beg);

        uint32_t byte_rate = sample_rate_.get() * 1 * 2;
        uint32_t data_chunk_size = samples_written_ * 2;
        uint32_t file_size = 36 + data_chunk_size;

        file_.write("RIFF", 4);
        write_bin(file_, file_size);
        file_.write("WAVE", 4);
        file_.write("fmt ", 4);
        write_bin(file_, (uint32_t)16);
        write_bin(file_, (uint16_t)1);
        write_bin(file_, (uint16_t)1);
        write_bin(file_, (uint32_t)sample_rate_.get());
        write_bin(file_, byte_rate);
        write_bin(file_, (uint16_t)2);
        write_bin(file_, (uint16_t)16);
        file_.write("data", 4);
        write_bin(file_, data_chunk_size);

        file_.seekp(current_pos); // Jump back to the end to seamlessly continue writing
    }
}

void AudioFileWriterOp::compute(holoscan::InputContext& input, holoscan::OutputContext& op_output, holoscan::ExecutionContext& context)
{
    // ALWAYS pop the message off the queue to prevent pipeline backpressure hangs
    auto in_message = input.receive<holoscan::gxf::Entity>("input");
    if (!in_message)
        return;

    if (recording_finished_)
        return;

    nvidia::gxf::Entity entity = static_cast<nvidia::gxf::Entity>(in_message.value());
    auto beam_tensor = entity.get<nvidia::gxf::Tensor>("beam_output");

    if (beam_tensor) {
        int num_samples = beam_tensor.value()->size() / sizeof(int32_t);

        uint32_t chunk_start = samples_seen_;
        uint32_t chunk_end = samples_seen_ + num_samples;
        samples_seen_ += num_samples;

        if (chunk_end <= start_sample_)
            return;

        if (host_buffer_.size() < num_samples)
            host_buffer_.resize(num_samples);
        cudaMemcpy(host_buffer_.data(), beam_tensor.value()->pointer(), num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);

        uint32_t write_start_idx = (chunk_start < start_sample_) ? (start_sample_ - chunk_start) : 0;
        uint32_t write_end_idx = (chunk_end > stop_sample_) ? (stop_sample_ - chunk_start) : num_samples;
        uint32_t write_samples = write_end_idx - write_start_idx;

        if (!recording_started_ && write_samples > 0) {
            recording_started_ = true;
            HSB_LOG_INFO("Recording STARTED (writing to {})...", file_path_.get());
        }

        // =====================================================================
        // SIGNAL CONDITIONING & FORMAT CONVERSION (RAW 32-bit to 16-bit PCM)
        // =====================================================================
        if (write_samples > 0) {
            std::vector<int16_t> out_buffer(write_samples);

            for (uint32_t i = 0; i < write_samples; ++i) {
                // 1. Full-Scale Normalization
                // Represents the exact, raw amplitude of the beamformer output [-1.0, 1.0]
                float normalized = (float)host_buffer_[write_start_idx + i] / 2147483648.0f;

                // 2. DC Blocker
                // Centers the raw waveform at 0.0 without altering its amplitude
                float filtered = normalized - dc_prev_x_ + 0.995f * dc_prev_y_;
                dc_prev_x_ = normalized;
                dc_prev_y_ = filtered;

                // 3. True 16-Bit PCM Projection (typically will require amplication to hear).
                // Maps the raw acoustic amplitude directly into the 16-bit space
                float pcm_float = filtered * 32767.0f;
                float clamped = fmaxf(-32768.0f, fminf(32767.0f, pcm_float));

                out_buffer[i] = static_cast<int16_t>(clamped);
            }

            file_.write(reinterpret_cast<const char*>(out_buffer.data()), write_samples * sizeof(int16_t));
            samples_written_ += write_samples;

            update_header();
        }

        if (samples_seen_ >= stop_sample_) {
            if (!recording_finished_) {
                recording_finished_ = true;
                HSB_LOG_INFO("Recording STOPPED at limit of {} seconds. File safely closed.", record_stop_.get());
                if (file_.is_open())
                    file_.close();
            }
        }
    }
}

void AudioFileWriterOp::stop()
{
    if (file_.is_open()) {
        update_header(); // Catch any final bytes if stopped early via Ctrl+C
        file_.close();
        if (samples_written_ > 0) {
            HSB_LOG_INFO("Saved complete {} second 16-bit track.", (float)samples_written_ / sample_rate_.get());
        }
    }
}

} // namespace hololink::operators
