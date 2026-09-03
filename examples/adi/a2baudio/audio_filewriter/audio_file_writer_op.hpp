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

#include <cstdint>
#include <fstream>
#include <holoscan/holoscan.hpp>
#include <string>
#include <vector>

namespace hololink::operators {

class AudioFileWriterOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(AudioFileWriterOp)

    AudioFileWriterOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void compute(holoscan::InputContext& input, holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) override;
    void stop() override;

private:
    void update_header();

    holoscan::Parameter<std::string> file_path_;
    holoscan::Parameter<int> sample_rate_;
    holoscan::Parameter<float> record_start_;
    holoscan::Parameter<float> record_stop_;

    std::ofstream file_;
    std::vector<int32_t> host_buffer_;

    uint32_t start_sample_ = 0;
    uint32_t stop_sample_ = 0;
    uint32_t samples_seen_ = 0;
    uint32_t samples_written_ = 0;

    float dc_prev_x_ = 0.0f;
    float dc_prev_y_ = 0.0f;

    bool recording_started_ = false;
    bool recording_finished_ = false;
};

} // namespace hololink::operators
