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
#ifndef SENSORS_ADI_IMU_HPP
#define SENSORS_ADI_IMU_HPP

#pragma once

#include <atomic>
#include <chrono>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/hololink.hpp>
#include <holoscan/holoscan.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hololink {
namespace ops {

    // ==============================================================================
    // HARDWARE REGISTER NAMESPACES (From Patch Baseline)
    // ==============================================================================

    namespace ADIS16505 {
        constexpr uint8_t XG_BIAS_LOW = 0x40;
        constexpr uint8_t YG_BIAS_LOW = 0x44;
        constexpr uint8_t ZG_BIAS_LOW = 0x48;
        constexpr uint8_t XA_BIAS_LOW = 0x4C;
        constexpr uint8_t YA_BIAS_LOW = 0x50;
        constexpr uint8_t ZA_BIAS_LOW = 0x54;
        constexpr uint8_t MSC_CTRL = 0x60;
        constexpr uint8_t UP_SCALE = 0x62;
        constexpr uint8_t DEC_RATE = 0x64;
        constexpr uint8_t GLOB_CMD = 0x68;

        constexpr uint16_t MSC_CTRL_SCALED_SYNC = 0x028B;
        constexpr uint16_t MSC_CTRL_INTERNAL = 0x0281;
    }

    namespace ADIS16607 {
        constexpr uint8_t SPI_BURST_CMD_85 = 0x85;
        constexpr uint16_t SPI_HALFDUPLEX_KEY = 0xB4B4;

        constexpr uint8_t XA_BIAS = 0x3B;
        constexpr uint8_t YA_BIAS = 0x3C;
        constexpr uint8_t ZA_BIAS = 0x3D;
        constexpr uint8_t XG_BIAS = 0x3E;
        constexpr uint8_t YG_BIAS = 0x3F;
        constexpr uint8_t ZG_BIAS = 0x40;

        constexpr uint8_t USER_GPIO_CFG1 = 0x2F;
        constexpr uint8_t SPI_CFG1 = 0x32;
        constexpr uint8_t SYNC_CFG = 0x33;
        constexpr uint8_t USER_DATA_CFG = 0x34;
        constexpr uint8_t MSC_CTRL = 0x39;
        constexpr uint8_t DEC_RATE = 0x3A;

        constexpr uint16_t GPIO_CFG1_SYNC_DR = 0x0241;
        constexpr uint16_t GPIO_CFG1_DR_ONLY = 0x0201;
        constexpr uint16_t USER_DATA_CFG_6DOF = 0xD03F;
        constexpr uint16_t MSC_CTRL_DEFAULT = 0x0100;
    }

    // ==============================================================================
    // DRIVER STRATEGY ARCHITECTURE
    // ==============================================================================

    struct ImuConfig {
        int sync_freq;
        int output_freq;
        int sync_pin;
        int32_t calib_gx, calib_gy, calib_gz;
        int32_t calib_ax, calib_ay, calib_az;
    };

    class AdisImuDriver {
    public:
        virtual ~AdisImuDriver() = default;
        virtual void initialize(const ImuConfig& config) = 0;
        virtual bool read_burst(std::string& out_imu_csv, std::string& out_temp_csv, uint16_t& last_cntr) = 0;
    };

    // Isolated ADIS16505-2 Driver (Full Duplex)
    class Adis16505Driver : public AdisImuDriver {
    public:
        Adis16505Driver(std::shared_ptr<hololink::Hololink::Spi> spi);
        void initialize(const ImuConfig& config) override;
        bool read_burst(std::string& out_imu_csv, std::string& out_temp_csv, uint16_t& last_cntr) override;

    private:
        void write_reg(uint8_t reg_addr, uint16_t value);
        std::shared_ptr<hololink::Hololink::Spi> spi_;
        std::vector<uint8_t> burst_cmd_;
    };

    // Isolated ADIS16607-2 Driver (Half Duplex)
    class Adis16607Driver : public AdisImuDriver {
    public:
        Adis16607Driver(std::shared_ptr<hololink::Hololink::Spi> spi);
        void initialize(const ImuConfig& config) override;
        bool read_burst(std::string& out_imu_csv, std::string& out_temp_csv, uint16_t& last_cntr) override;

    private:
        void write_16607_hd(uint8_t reg_addr, uint16_t value);
        std::shared_ptr<hololink::Hololink::Spi> spi_;
        std::vector<uint8_t> burst_cmd_;
    };

    // ==============================================================================
    // OPERATOR CLASS DEFINITION
    // ==============================================================================

    class ImuHardwareOp : public holoscan::Operator {
    public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(ImuHardwareOp)
        ~ImuHardwareOp();

        void initialize() override;
        void start() override;
        void stop() override;
        void setup(holoscan::OperatorSpec& spec) override;

        void compute(holoscan::InputContext& op_input,
            holoscan::OutputContext& op_output,
            holoscan::ExecutionContext& context) override;

    private:
        void validate_configuration();
        void hardware_loop();

        holoscan::Parameter<std::string> imu_model_;
        holoscan::Parameter<std::string> hsb_ip_;
        holoscan::Parameter<int> sync_freq_;
        holoscan::Parameter<int> output_freq_;
        holoscan::Parameter<int> spi_port_;
        holoscan::Parameter<int> spi_cs_;
        holoscan::Parameter<int> spi_div_;
        holoscan::Parameter<int> spi_cpol_;
        holoscan::Parameter<int> spi_cpha_;

        holoscan::Parameter<int> reset_pin_;
        holoscan::Parameter<int> dr_pin_;
        holoscan::Parameter<int> sync_pin_;

        holoscan::Parameter<int32_t> calib_gx_;
        holoscan::Parameter<int32_t> calib_gy_;
        holoscan::Parameter<int32_t> calib_gz_;
        holoscan::Parameter<int32_t> calib_ax_;
        holoscan::Parameter<int32_t> calib_ay_;
        holoscan::Parameter<int32_t> calib_az_;

        std::shared_ptr<hololink::DataChannel> data_channel_;
        std::shared_ptr<hololink::Hololink::Spi> spi_;
        std::shared_ptr<hololink::Hololink::GPIO> gpio_;

        // The Polymorphic Driver Instance
        std::unique_ptr<AdisImuDriver> driver_;
        ImuConfig current_config_;

        std::thread worker_thread_;
        std::atomic<bool> keep_running_ { false };
        std::mutex data_mutex_;

        int sample_count_ = 0;
        int duplicate_read_count_ = 0;
        double last_report_time_ = 0.0;

        std::string latest_imu_csv_;
        std::string latest_temp_csv_;
        uint16_t thread_last_cntr_ = 0;
    };

} // namespace ops
} // namespace hololink

#endif // SENSORS_ADI_IMU_HPP
