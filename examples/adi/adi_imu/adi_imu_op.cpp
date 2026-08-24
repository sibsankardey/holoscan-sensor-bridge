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
#include "adi_imu_op.hpp"

#include <hololink/core/enumerator.hpp>
#include <hololink/core/timeout.hpp>
#include <holoscan/holoscan.hpp>

#include <cmath>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace hololink {
namespace ops {

    // =========================================================================
    // ISOLATED ADIS16505-2 DRIVER IMPLEMENTATION
    // =========================================================================

    Adis16505Driver::Adis16505Driver(std::shared_ptr<hololink::Hololink::Spi> spi)
        : spi_(spi)
        , burst_cmd_({ ADIS16505::GLOB_CMD, 0x00 })
    {
    }

    void Adis16505Driver::write_reg(uint8_t reg_addr, uint16_t value)
    {
        auto timeout = std::make_shared<hololink::Timeout>(0.5, 0.1);
        uint8_t low_addr = 0x80 | reg_addr;
        std::vector<uint8_t> low_cmd = { low_addr, static_cast<uint8_t>(value & 0xFF) };
        spi_->spi_transaction(low_cmd, {}, 0, timeout);
        usleep(100);

        uint8_t high_addr = 0x80 | (reg_addr + 1);
        std::vector<uint8_t> high_cmd = { high_addr, static_cast<uint8_t>((value >> 8) & 0xFF) };
        spi_->spi_transaction(high_cmd, {}, 0, timeout);
        usleep(100);
    }

    void Adis16505Driver::initialize(const ImuConfig& config)
    {
        HOLOSCAN_LOG_INFO("Executing ADIS16505-2 Initialization...");

        if (config.calib_gx != 0 || config.calib_ax != 0 || config.calib_gy != 0) {
            HOLOSCAN_LOG_INFO("Injecting Software Calibration Flags into Hardware Registers...");
            auto write_bias_32 = [&](uint8_t low_addr, int32_t val) {
                if (val == 0)
                    return;
                uint16_t low_word = val & 0xFFFF;
                uint16_t high_word = (val >> 16) & 0xFFFF;
                write_reg(low_addr, low_word);
                usleep(100);
                write_reg(low_addr + 2, high_word);
            };

            write_bias_32(ADIS16505::XG_BIAS_LOW, config.calib_gx);
            write_bias_32(ADIS16505::YG_BIAS_LOW, config.calib_gy);
            write_bias_32(ADIS16505::ZG_BIAS_LOW, config.calib_gz);
            write_bias_32(ADIS16505::XA_BIAS_LOW, config.calib_ax);
            write_bias_32(ADIS16505::YA_BIAS_LOW, config.calib_ay);
            write_bias_32(ADIS16505::ZA_BIAS_LOW, config.calib_az);
        }

        if (config.sync_pin > 0 && config.sync_pin < 100) {
            int output_multiplier = std::round(2000.0 / config.output_freq);
            int internal_freq = config.output_freq * output_multiplier;
            uint8_t up_scale_val = internal_freq / config.sync_freq;
            uint8_t dec_rate_val = output_multiplier - 1;

            write_reg(ADIS16505::UP_SCALE, up_scale_val);
            usleep(20000);
            write_reg(ADIS16505::DEC_RATE, dec_rate_val);
            usleep(20000);
            write_reg(ADIS16505::MSC_CTRL, ADIS16505::MSC_CTRL_SCALED_SYNC);
            usleep(20000);
        } else {
            int dec_rate_val = std::round(2000.0 / config.output_freq) - 1;
            if (dec_rate_val < 0)
                dec_rate_val = 0;
            write_reg(ADIS16505::DEC_RATE, dec_rate_val);
            usleep(20000);
            write_reg(ADIS16505::MSC_CTRL, ADIS16505::MSC_CTRL_INTERNAL);
            usleep(20000);
        }
    }

    bool Adis16505Driver::read_burst(std::string& out_imu_csv, std::string& out_temp_csv, uint16_t& last_cntr)
    {
        auto timeout = std::make_shared<hololink::Timeout>(0.1, 0.01);
        std::vector<uint8_t> rx_data = spi_->spi_transaction(burst_cmd_, {}, 32, timeout);

        if (rx_data.size() == 34) {
            auto get_word = [&rx_data](size_t offset) -> uint16_t {
                return (rx_data[offset] << 8) | rx_data[offset + 1];
            };

            uint16_t current_cntr = get_word(30);

            if (current_cntr != last_cntr) {
                auto now = std::chrono::system_clock::now();
                double now_time = std::chrono::duration<double>(now.time_since_epoch()).count();

                uint32_t gx_u = (static_cast<uint32_t>(get_word(6)) << 16) | get_word(4);
                uint32_t gy_u = (static_cast<uint32_t>(get_word(10)) << 16) | get_word(8);
                uint32_t gz_u = (static_cast<uint32_t>(get_word(14)) << 16) | get_word(12);
                uint32_t ax_u = (static_cast<uint32_t>(get_word(18)) << 16) | get_word(16);
                uint32_t ay_u = (static_cast<uint32_t>(get_word(22)) << 16) | get_word(20);
                uint32_t az_u = (static_cast<uint32_t>(get_word(26)) << 16) | get_word(24);
                int16_t temp_raw = static_cast<int16_t>(get_word(28));

                out_imu_csv = std::to_string(now_time) + "," + std::to_string(static_cast<int32_t>(gx_u)) + "," + std::to_string(static_cast<int32_t>(gy_u)) + "," + std::to_string(static_cast<int32_t>(gz_u)) + "," + std::to_string(static_cast<int32_t>(ax_u)) + "," + std::to_string(static_cast<int32_t>(ay_u)) + "," + std::to_string(static_cast<int32_t>(az_u));
                out_temp_csv = std::to_string(now_time) + "," + std::to_string(temp_raw);

                last_cntr = current_cntr;
                return true;
            }
        }
        return false;
    }

    // =========================================================================
    // ISOLATED ADIS16607-2 DRIVER IMPLEMENTATION
    // =========================================================================

    Adis16607Driver::Adis16607Driver(std::shared_ptr<hololink::Hololink::Spi> spi)
        : spi_(spi)
    {
        burst_cmd_ = std::vector<uint8_t>(34, 0x00);
        burst_cmd_[0] = ADIS16607::SPI_BURST_CMD_85;
    }

    void Adis16607Driver::write_16607_hd(uint8_t reg_addr, uint16_t value)
    {
        auto timeout = std::make_shared<hololink::Timeout>(1.0, 0.1);
        std::vector<uint8_t> tx = {
            static_cast<uint8_t>(reg_addr & 0x7F),
            static_cast<uint8_t>((value >> 8) & 0xFF),
            static_cast<uint8_t>(value & 0xFF)
        };
        spi_->spi_transaction(tx, {}, 0, timeout);
        usleep(50000);
    }

    void Adis16607Driver::initialize(const ImuConfig& config)
    {
        HOLOSCAN_LOG_INFO("Executing ADIS16607-2 Initialization via Standardized API...");

        if (config.calib_gx != 0 || config.calib_ax != 0 || config.calib_gy != 0) {
            HOLOSCAN_LOG_INFO("Injecting Software Calibration Flags into Hardware Registers...");
            auto write_bias_16 = [&](uint8_t addr, int32_t val) {
                if (val == 0)
                    return;
                int16_t clamped_val = static_cast<int16_t>(std::max(-32768, std::min(32767, val)));
                write_16607_hd(addr, static_cast<uint16_t>(clamped_val));
            };

            write_bias_16(ADIS16607::XA_BIAS, config.calib_ax);
            write_bias_16(ADIS16607::YA_BIAS, config.calib_ay);
            write_bias_16(ADIS16607::ZA_BIAS, config.calib_az);
            write_bias_16(ADIS16607::XG_BIAS, config.calib_gx);
            write_bias_16(ADIS16607::YG_BIAS, config.calib_gy);
            write_bias_16(ADIS16607::ZG_BIAS, config.calib_gz);
        }

        if (config.sync_pin > 0 && config.sync_pin < 100) {
            write_16607_hd(ADIS16607::SPI_CFG1, ADIS16607::SPI_HALFDUPLEX_KEY);
            write_16607_hd(ADIS16607::USER_GPIO_CFG1, ADIS16607::GPIO_CFG1_SYNC_DR);

            int sync_hz = std::round(config.sync_freq);
            int out_hz = std::round(config.output_freq);
            auto compute_gcd = [](int a, int b) { while (b != 0) { int t = b; b = a % b; a = t; } return a; };
            int base_hz = (sync_hz * out_hz) / compute_gcd(sync_hz, out_hz);

            int ideal_native_clock = 4000;
            int pll_multiplier = std::max(1, static_cast<int>(std::round(static_cast<double>(ideal_native_clock) / base_hz)));
            uint16_t target_internal_hz = base_hz * pll_multiplier;
            if (target_internal_hz > 9000)
                target_internal_hz = (9000 / base_hz) * base_hz;

            uint16_t sync_scale_factor = target_internal_hz / sync_hz;
            uint16_t user_sync_val = 0x8000 | (sync_scale_factor & 0x7FFF);
            uint16_t dec_rate_16607 = (target_internal_hz / out_hz) - 1;

            write_16607_hd(ADIS16607::SYNC_CFG, user_sync_val);
            write_16607_hd(ADIS16607::DEC_RATE, dec_rate_16607);
            write_16607_hd(ADIS16607::USER_DATA_CFG, ADIS16607::USER_DATA_CFG_6DOF);
        } else {
            int dec_rate_val = std::round(2000.0 / config.output_freq) - 1;
            if (dec_rate_val < 0)
                dec_rate_val = 0;

            write_16607_hd(ADIS16607::SPI_CFG1, ADIS16607::SPI_HALFDUPLEX_KEY);
            write_16607_hd(ADIS16607::USER_GPIO_CFG1, ADIS16607::GPIO_CFG1_DR_ONLY);
            write_16607_hd(ADIS16607::SYNC_CFG, 0x0000);
            write_16607_hd(ADIS16607::DEC_RATE, dec_rate_val);
            write_16607_hd(ADIS16607::USER_DATA_CFG, ADIS16607::USER_DATA_CFG_6DOF);
        }
    }

    bool Adis16607Driver::read_burst(std::string& out_imu_csv, std::string& out_temp_csv, uint16_t& last_cntr)
    {
        auto timeout = std::make_shared<hololink::Timeout>(0.1, 0.01);

        std::vector<uint8_t> rx = spi_->spi_transaction(burst_cmd_, {}, 0, timeout);

        if (rx.size() == 34) {
            auto get_word = [&rx](size_t offset) -> uint16_t {
                return (static_cast<uint16_t>(rx[2 + offset]) << 8) | rx[2 + offset + 1];
            };

            uint16_t calc_chksum = 0;
            for (size_t offset = 0; offset <= 28; offset += 2) {
                calc_chksum += get_word(offset);
            }

            if (calc_chksum == get_word(30)) {
                uint16_t current_cntr = get_word(28);

                if (current_cntr != last_cntr) {
                    auto now = std::chrono::system_clock::now();
                    double now_time = std::chrono::duration<double>(now.time_since_epoch()).count();

                    auto parse_32to24 = [](uint16_t high, uint16_t low) -> int32_t {
                        int32_t raw_32 = static_cast<int32_t>((static_cast<uint32_t>(high) << 16) | low);
                        return raw_32 >> 8;
                    };

                    int32_t ax_u = parse_32to24(get_word(2), get_word(4));
                    int32_t ay_u = parse_32to24(get_word(6), get_word(8));
                    int32_t az_u = parse_32to24(get_word(10), get_word(12));

                    int32_t gx_u = parse_32to24(get_word(14), get_word(16));
                    int32_t gy_u = parse_32to24(get_word(18), get_word(20));
                    int32_t gz_u = parse_32to24(get_word(22), get_word(24));

                    int16_t temp_raw = static_cast<int16_t>(get_word(26));

                    out_imu_csv = std::to_string(now_time) + "," + std::to_string(gx_u) + "," + std::to_string(gy_u) + "," + std::to_string(gz_u) + "," + std::to_string(ax_u) + "," + std::to_string(ay_u) + "," + std::to_string(az_u);
                    out_temp_csv = std::to_string(now_time) + "," + std::to_string(temp_raw);

                    last_cntr = current_cntr;
                    return true;
                }
            }
        }
        return false;
    }

    // =========================================================================
    // HOLOSCAN OPERATOR IMPLEMENTATION (Hardware Agnostic)
    // =========================================================================

    ImuHardwareOp::~ImuHardwareOp()
    {
        keep_running_ = false;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    void ImuHardwareOp::setup(holoscan::OperatorSpec& spec)
    {
        spec.param(imu_model_, "imu_model", "IMU Model", "Target IMU hardware", std::string("ADIS16505-2"));
        spec.param(hsb_ip_, "hsb_ip", "HSB IP", "IP of the Holoscan Sensor Bridge", std::string("192.168.0.2"));
        spec.param(sync_freq_, "sync_freq", "Sync Frequency", "External sync frequency (Hz)", 120);
        spec.param(output_freq_, "output_freq", "Output Frequency", "Desired output rate (Hz)", 240);
        spec.param(spi_port_, "spi_port", "SPI Port", "SPI Port on HSB", 2);
        spec.param(spi_cs_, "spi_cs", "SPI CS", "SPI Chip Select", 0);
        spec.param(spi_div_, "spi_div", "SPI Divisor", "SPI Clock Divisor", 10);
        spec.param(spi_cpol_, "spi_cpol", "SPI CPOL", "Clock Polarity", 1);
        spec.param(spi_cpha_, "spi_cpha", "SPI CPHA", "Clock Phase", 1);

        spec.param(reset_pin_, "reset_pin", "Reset Pin", "GPIO Pin for Reset", 9);
        spec.param(dr_pin_, "dr_pin", "DR Pin", "GPIO Pin for DR", 8);
        spec.param(sync_pin_, "sync_pin", "Sync Pin", "GPIO Pin for Sync (-1 to ignore)", -1);

        spec.param(calib_gx_, "calib_gx", "Calib GX", "Gyro X Bias", 0);
        spec.param(calib_gy_, "calib_gy", "Calib GY", "Gyro Y Bias", 0);
        spec.param(calib_gz_, "calib_gz", "Calib GZ", "Gyro Z Bias", 0);
        spec.param(calib_ax_, "calib_ax", "Calib AX", "Accel X Bias", 0);
        spec.param(calib_ay_, "calib_ay", "Calib AY", "Accel Y Bias", 0);
        spec.param(calib_az_, "calib_az", "Calib AZ", "Accel Z Bias", 0);

        spec.output<std::string>("imu_data");
        spec.output<std::string>("temp_data");
    }

    void ImuHardwareOp::validate_configuration()
    {
        if (sync_freq_.get() < 1 || sync_freq_.get() > 128) {
            HOLOSCAN_LOG_ERROR("FATAL: Sync frequency must be 1-128 Hz.");
            throw std::runtime_error("Invalid Sync Frequency");
        }
    }

    void ImuHardwareOp::initialize()
    {
        holoscan::Operator::initialize();
        validate_configuration();

        current_config_.sync_freq = sync_freq_.get();
        current_config_.output_freq = output_freq_.get();
        current_config_.sync_pin = sync_pin_.get();
        current_config_.calib_gx = calib_gx_.get();
        current_config_.calib_gy = calib_gy_.get();
        current_config_.calib_gz = calib_gz_.get();
        current_config_.calib_ax = calib_ax_.get();
        current_config_.calib_ay = calib_ay_.get();
        current_config_.calib_az = calib_az_.get();

        HOLOSCAN_LOG_INFO("Connecting to Holoscan Sensor Bridge at {}...", hsb_ip_.get());

        auto metadata = hololink::Enumerator::find_channel(hsb_ip_.get());
        if (metadata.empty()) {
            HOLOSCAN_LOG_ERROR("HSB not found.");
            throw std::runtime_error("HSB Connection Failed");
        }

        data_channel_ = std::make_shared<hololink::DataChannel>(metadata);
        auto hl = data_channel_->hololink();
        hl->start();

        gpio_ = hl->get_gpio(metadata);
        spi_ = hl->get_spi(spi_port_.get(), spi_cs_.get(), spi_div_.get(), spi_cpol_.get(), spi_cpha_.get());

        HOLOSCAN_LOG_INFO("Performing Hardware Reset on Pin {}...", reset_pin_.get());
        gpio_->set_direction(reset_pin_.get(), 0);
        gpio_->set_value(reset_pin_.get(), 0);
        usleep(100000);
        gpio_->set_value(reset_pin_.get(), 1);
        usleep(1000000);

        gpio_->set_direction(dr_pin_.get(), 1);

        std::string model = imu_model_.get();
        HOLOSCAN_LOG_INFO("Initializing Hardware for Model: {}", model);

        if (model == "ADIS16505-2") {
            driver_ = std::make_unique<Adis16505Driver>(spi_);
        } else if (model == "ADIS16607-2") {
            driver_ = std::make_unique<Adis16607Driver>(spi_);
        } else {
            HOLOSCAN_LOG_ERROR("UNKNOWN IMU MODEL: {}", model);
            throw std::runtime_error("Invalid IMU Model");
        }

        driver_->initialize(current_config_);

        last_report_time_ = holoscan::get_current_time_us() / 1000000.0;
        HOLOSCAN_LOG_INFO("C++ IMU Operator Initialized Successfully.");
    }

    void ImuHardwareOp::start()
    {
        keep_running_ = true;
        worker_thread_ = std::thread(&ImuHardwareOp::hardware_loop, this);
    }

    void ImuHardwareOp::hardware_loop()
    {
        pthread_t this_thread = pthread_self();
        struct sched_param params;
        params.sched_priority = 90;

        if (pthread_setschedparam(this_thread, SCHED_FIFO, &params) != 0) {
            HOLOSCAN_LOG_WARN("Failed to set SCHED_FIFO. Network jitter may occur.");
        }

        double poll_rate_hz = output_freq_.get() * 2.5;
        long period_ns = static_cast<long>((1.0 / poll_rate_hz) * 1e9);

        HOLOSCAN_LOG_INFO("Starting SPI Virtual-DR Polling Loop at ~{:.0f} Hz...", poll_rate_hz);

        struct timespec next_poll_time;
        clock_gettime(CLOCK_MONOTONIC, &next_poll_time);

        std::string temp_imu_csv, temp_temp_csv;

        while (keep_running_) {
            bool frame_was_fresh = false;

            try {
                frame_was_fresh = driver_->read_burst(temp_imu_csv, temp_temp_csv, thread_last_cntr_);

                if (frame_was_fresh) {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    latest_imu_csv_ = temp_imu_csv;
                    latest_temp_csv_ = temp_temp_csv;
                }
            } catch (const std::exception& e) {
                std::string err_msg = e.what();
                if (err_msg.find("RESPONSE_SEQUENCE_CHECK_FAIL") == std::string::npos && err_msg.find("Timeout") == std::string::npos) {
                    HOLOSCAN_LOG_WARN("Network/SPI Error: {}", err_msg);
                }
            }

            if (frame_was_fresh) {
                sample_count_++;
            } else {
                duplicate_read_count_++;
            }

            struct timespec now_ts;
            clock_gettime(CLOCK_MONOTONIC, &now_ts);
            long diff_ns = (now_ts.tv_sec - next_poll_time.tv_sec) * 1000000000L + (now_ts.tv_nsec - next_poll_time.tv_nsec);
            if (diff_ns > 10000000L) {
                next_poll_time = now_ts;
            }

            next_poll_time.tv_nsec += period_ns;
            while (next_poll_time.tv_nsec >= 1000000000) {
                next_poll_time.tv_nsec -= 1000000000;
                next_poll_time.tv_sec++;
            }
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_poll_time, NULL);
        }
    }

    void ImuHardwareOp::compute(holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context)
    {

        double now_time = holoscan::get_current_time_us() / 1000000.0;
        std::string imu_to_send;
        std::string temp_to_send;

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            imu_to_send = latest_imu_csv_;
            temp_to_send = latest_temp_csv_;

            latest_imu_csv_.clear();
            latest_temp_csv_.clear();
        }

        if (!imu_to_send.empty()) {
            op_output.emit(imu_to_send, "imu_data");
            op_output.emit(temp_to_send, "temp_data");
        }

        if (now_time - last_report_time_ >= 5.0) {
            double hz = sample_count_ / (now_time - last_report_time_);
            HOLOSCAN_LOG_INFO("[{}] Output: {:.1f} Hz", name(), hz); // duplicate_read_count_);
            sample_count_ = 0;
            duplicate_read_count_ = 0;
            last_report_time_ = now_time;
        }
    }

    void ImuHardwareOp::stop()
    {
        keep_running_ = false;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

} // namespace ops
} // namespace hololink
