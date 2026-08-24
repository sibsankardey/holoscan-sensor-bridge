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
#include "../adi_imu_op.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::shared_ptr<hololink::ops::ImuHardwareOp> create_imu_operator(
    holoscan::Fragment* fragment, const std::string& name,
    const std::string& imu_model, const std::string& hsb_ip,
    int sync_freq, int output_freq,
    int spi_port, int spi_cs, int spi_div, int spi_cpol, int spi_cpha,
    int reset_pin, int dr_pin, int sync_pin,
    int32_t calib_gx, int32_t calib_gy, int32_t calib_gz,
    int32_t calib_ax, int32_t calib_ay, int32_t calib_az)
{
    return fragment->make_operator<hololink::ops::ImuHardwareOp>(
        name,
        holoscan::Arg("imu_model", imu_model),
        holoscan::Arg("hsb_ip", hsb_ip),
        holoscan::Arg("sync_freq", sync_freq),
        holoscan::Arg("output_freq", output_freq),
        holoscan::Arg("spi_port", spi_port),
        holoscan::Arg("spi_cs", spi_cs),
        holoscan::Arg("spi_div", spi_div),
        holoscan::Arg("spi_cpol", spi_cpol),
        holoscan::Arg("spi_cpha", spi_cpha),
        holoscan::Arg("reset_pin", reset_pin),
        holoscan::Arg("dr_pin", dr_pin),
        holoscan::Arg("sync_pin", sync_pin),
        holoscan::Arg("calib_gx", calib_gx),
        holoscan::Arg("calib_gy", calib_gy),
        holoscan::Arg("calib_gz", calib_gz),
        holoscan::Arg("calib_ax", calib_ax),
        holoscan::Arg("calib_ay", calib_ay),
        holoscan::Arg("calib_az", calib_az));
}

PYBIND11_MODULE(_adi_imu_op, m)
{
    // 1. Register the class type so Pybind11 knows how to wrap the return value
    py::class_<hololink::ops::ImuHardwareOp,
        holoscan::Operator,
        std::shared_ptr<hololink::ops::ImuHardwareOp>>(m, "ImuHardwareOp");

    // 2. Register the factory function
    m.def("create_imu_operator", &create_imu_operator, "Create IMU Hardware Operator");
}
