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

#ifndef SENSORS_ADCAM_CALIBRATION_HPP
#define SENSORS_ADCAM_CALIBRATION_HPP

#include <cstdint>
#include <vector>

namespace hololink::sensors {

// ---------------------------------------------------------------------------
// Calibration data served by the ADSD3500 out of the module CCB.
//
// The ADSD3500 firmware parses the CCB stored in the module flash and exposes
// the per-mode results through two burst-mode read-payload commands:
//   0x01 -> 56 bytes of camera intrinsics
//   0x02 -> 32 bytes of dealias / sensor-geometry parameters
// Both payloads are raw little-endian struct images, so the layouts below must
// stay byte-compatible with CameraIntrinsics / TofiXYZDealiasData in the ADI
// ToF SDK (sdk/common/adi/tofi/tofi_camera_intrinsics.h).
// ---------------------------------------------------------------------------

struct AdcamCameraIntrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
    float codx;
    float cody;
    float k1;
    float k2;
    float k3;
    float k4;
    float k5;
    float k6;
    float p2;
    float p1;
};
static_assert(sizeof(AdcamCameraIntrinsics) == 56,
    "AdcamCameraIntrinsics must match the 56-byte ADSD3500 payload");

struct AdcamDealiasData {
    int32_t n_rows;
    int32_t n_cols;
    uint8_t n_freqs;
    uint8_t row_bin_factor;
    uint8_t col_bin_factor;
    uint16_t n_offset_rows;
    uint16_t n_offset_cols;
    uint16_t n_sensor_rows;
    uint16_t n_sensor_cols;
    uint8_t freq_index[3];
    uint16_t freq[3];
};
static_assert(sizeof(AdcamDealiasData) == 32,
    "AdcamDealiasData must match the 32-byte ADSD3500 payload");

struct AdcamCalibration {
    AdcamDealiasData dealias;
    AdcamCameraIntrinsics intrinsics;
};

/// Build the per-pixel scale that turns the ADSD3500 radial depth into Cartesian Z:
///
///     Z(x, y) = radial(x, y) * table[y * out_cols + x]
///
/// Entries are 0 for pixels that fall outside the radius where the distortion
/// model can be inverted; those pixels have no valid Z.
///
/// Port of Algorithms::GenerateXYZTables() from the ADI ToF SDK, reduced to the
/// Z component. Returns an empty vector when the calibration is inconsistent
/// with the requested output geometry.
std::vector<float> adcam_generate_z_table(const AdcamCalibration& calibration,
    int out_rows,
    int out_cols,
    int undistort_iterations = 20);

} // namespace hololink::sensors

#endif /* SENSORS_ADCAM_CALIBRATION_HPP */
