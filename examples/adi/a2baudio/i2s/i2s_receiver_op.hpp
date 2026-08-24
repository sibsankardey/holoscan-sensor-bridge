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

#include <hololink/core/data_channel.hpp>
#include <hololink/operators/linux_receiver/linux_receiver_op.hpp>
#include <holoscan/holoscan.hpp>
#include <memory>

namespace hololink::operators {

class I2sReceiverOp : public hololink::operators::LinuxReceiverOp {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(I2sReceiverOp, hololink::operators::LinuxReceiverOp);

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;

    void compute(holoscan::InputContext& input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

    void start_hardware();
    void stop_hardware();

private:
    holoscan::Parameter<hololink::DataChannel*> i2s_channel_;
    holoscan::Parameter<uint32_t> i2s_address_;
    holoscan::Parameter<uint32_t> chunk_size_;
    holoscan::Parameter<uint32_t> i2s_cfg_value_;

    std::shared_ptr<hololink::Hololink::I2s> i2s_;
    bool hardware_started_ = false;
};

} // namespace hololink::operators
