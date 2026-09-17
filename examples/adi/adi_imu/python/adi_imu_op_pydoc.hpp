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
#pragma once
#include <string>

namespace hololink {
namespace ops {
    namespace doc {

        namespace ImuDataStruct {
            constexpr const char* doc_ImuDataStruct = R"doc(
Raw IMU Data Payload.

Contains the hardware timestamp and the raw, unscaled 32-bit integers 
for the X, Y, and Z axes of the gyroscope and accelerometer.
)doc";
        } // namespace ImuDataStruct

        namespace TempDataStruct {
            constexpr const char* doc_TempDataStruct = R"doc(
Raw Temperature Payload.

Contains the hardware timestamp and the raw 16-bit integer for the IMU temperature.
)doc";
        } // namespace TempDataStruct

        namespace ImuHardwareOp {
            constexpr const char* doc_ImuHardwareOp = R"doc(
High-Speed SPI Polling Operator for Analog Devices IMUs.

This operator interfaces with the ADIS1650x series IMUs over SPI using the 
Holoscan Sensor Bridge (HSB). It natively handles phase-locked loop (PLL) 
synchronization and executes high-frequency network over-polling in C++ 
to eliminate GPIO interrupt latency.

Outputs:
    imu_data (ImuDataStruct): The raw 32-bit burst payload.
    temp_data (TempDataStruct): The raw 16-bit temperature payload.
)doc";

            constexpr const char* doc_setup = R"doc(
Define the operator specification.
)doc";
        } // namespace ImuHardwareOp

    } // namespace doc
} // namespace ops
} // namespace hololink
