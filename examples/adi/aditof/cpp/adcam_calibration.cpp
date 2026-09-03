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

#include "adcam_calibration.hpp"

#include <algorithm>
#include <cmath>

#include <hololink/core/logging.hpp>
#include <holoscan/logger/logger.hpp>

namespace hololink::sensors {
namespace {

    /// Iteratively invert the Brown-Conrady model, mapping pixel coordinates to
    /// normalized (undistorted) camera coordinates. Port of UndistortPoints()
    /// from the ADI ToF SDK; the SDK zeroes the thin-prism terms, so they are
    /// dropped here.
    void undistort_points(std::vector<float>& x_points,
        std::vector<float>& y_points,
        const AdcamCameraIntrinsics& intrinsics,
        int max_count,
        uint8_t row_bin_factor,
        uint8_t col_bin_factor)
    {
        const double k1 = intrinsics.k1;
        const double k2 = intrinsics.k2;
        const double k3 = intrinsics.k3;
        const double k4 = intrinsics.k4;
        const double k5 = intrinsics.k5;
        const double k6 = intrinsics.k6;
        const double p1 = intrinsics.p1;
        const double p2 = intrinsics.p2;

        // The SDK scales the horizontal terms by row_bin_factor and the vertical
        // terms by col_bin_factor; keep that pairing so results match the SDK.
        const double fx = intrinsics.fx / row_bin_factor;
        const double fy = intrinsics.fy / col_bin_factor;
        const double cx = intrinsics.cx / row_bin_factor;
        const double cy = intrinsics.cy / col_bin_factor;
        const double ifx = 1.0 / fx;
        const double ify = 1.0 / fy;

        for (size_t i = 0; i < x_points.size(); ++i) {
            const double u = x_points[i];
            const double v = y_points[i];

            double x = (u - cx) * ifx;
            double y = (v - cy) * ify;
            const double x0 = x;
            const double y0 = y;

            for (int iteration = 0; iteration < max_count; ++iteration) {
                const double r2 = x * x + y * y;
                const double icdist = (1 + ((k6 * r2 + k5) * r2 + k4) * r2)
                    / (1 + ((k3 * r2 + k2) * r2 + k1) * r2);
                if (icdist < 0) {
                    x = (u - cx) * ifx;
                    y = (v - cy) * ify;
                    break;
                }
                const double delta_x = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
                const double delta_y = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
                x = (x0 - delta_x) * icdist;
                y = (y0 - delta_y) * icdist;
            }

            x_points[i] = static_cast<float>(x);
            y_points[i] = static_cast<float>(y);
        }
    }

} // namespace

std::vector<float> adcam_generate_z_table(const AdcamCalibration& calibration,
    int out_rows,
    int out_cols,
    int undistort_iterations)
{
    const AdcamDealiasData& dealias = calibration.dealias;

    if ((dealias.row_bin_factor == 0) || (dealias.col_bin_factor == 0)
        || (dealias.n_sensor_rows == 0) || (dealias.n_sensor_cols == 0)) {
        HOLOSCAN_LOG_ERROR("adcam_generate_z_table: invalid dealias data "
                           "(sensor={}x{} bin={}x{})",
            dealias.n_sensor_rows, dealias.n_sensor_cols,
            dealias.row_bin_factor, dealias.col_bin_factor);
        return {};
    }

    const int n_rows = dealias.n_sensor_rows / dealias.row_bin_factor;
    const int n_cols = dealias.n_sensor_cols / dealias.col_bin_factor;

    if ((out_rows <= 0) || (out_cols <= 0)
        || (dealias.n_offset_rows + out_rows > n_rows)
        || (dealias.n_offset_cols + out_cols > n_cols)) {
        HOLOSCAN_LOG_ERROR("adcam_generate_z_table: requested {}x{} output at "
                           "offset {},{} does not fit the binned sensor {}x{}",
            out_rows, out_cols, dealias.n_offset_rows, dealias.n_offset_cols,
            n_rows, n_cols);
        return {};
    }

    const size_t binned_count = static_cast<size_t>(n_rows) * n_cols;
    std::vector<float> x_points(binned_count);
    std::vector<float> y_points(binned_count);
    for (int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {
            const size_t index = static_cast<size_t>(row) * n_cols + col;
            x_points[index] = static_cast<float>(col);
            y_points[index] = static_cast<float>(row);
        }
    }

    undistort_points(x_points, y_points, calibration.intrinsics,
        undistort_iterations, dealias.row_bin_factor, dealias.col_bin_factor);

    // Ray length relative to the optical axis: radial = z * ray_length.
    std::vector<float> ray_length(binned_count);
    const float cx = calibration.intrinsics.cx / dealias.row_bin_factor;
    const float cy = calibration.intrinsics.cy / dealias.col_bin_factor;
    float r_min = std::sqrt(static_cast<float>(n_rows) * n_rows
        + static_cast<float>(n_cols) * n_cols);

    for (int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {
            const size_t index = static_cast<size_t>(row) * n_cols + col;
            const float x = x_points[index];
            const float y = y_points[index];
            ray_length[index] = std::sqrt(x * x + y * y + 1.0f);

            if (std::isnan(x) || std::isnan(y) || std::isnan(ray_length[index])
                || (ray_length[index] == 0.0f)) {
                const float dx = static_cast<float>(col) - cx;
                const float dy = static_cast<float>(row) - cy;
                r_min = std::min(r_min, std::sqrt(dx * dx + dy * dy));
            }
        }
    }
    r_min -= 2.0f; // buffer around the innermost invalid pixel

    std::vector<float> z_table(static_cast<size_t>(out_rows) * out_cols, 0.0f);
    for (int row = 0; row < out_rows; ++row) {
        for (int col = 0; col < out_cols; ++col) {
            const size_t source = static_cast<size_t>(row + dealias.n_offset_rows) * n_cols
                + col + dealias.n_offset_cols;
            const float dx = static_cast<float>(col + dealias.n_offset_cols) - cx;
            const float dy = static_cast<float>(row + dealias.n_offset_rows) - cy;
            if (std::sqrt(dx * dx + dy * dy) >= r_min) {
                continue; // outside the valid radius
            }
            if (ray_length[source] != 0.0f) {
                z_table[static_cast<size_t>(row) * out_cols + col] = 1.0f / ray_length[source];
            }
        }
    }

    HOLOSCAN_LOG_INFO("Generated {}x{} radial->Z table from binned sensor {}x{} "
                      "(offset {},{}, valid radius {:.1f} px)",
        out_rows, out_cols, n_rows, n_cols, dealias.n_offset_rows,
        dealias.n_offset_cols, r_min);

    return z_table;
}

} // namespace hololink::sensors
