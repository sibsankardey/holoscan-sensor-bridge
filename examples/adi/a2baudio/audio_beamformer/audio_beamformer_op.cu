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
#include "audio_beamformer_op.hpp"
#include <gxf/std/tensor.hpp>
#include <hololink/core/logging_internal.hpp>
#include <device_launch_parameters.h>
#include <math.h>
#include <vector>

namespace hololink::operators {

// =====================================================================
// GLOBAL CONSTANT MEMORY
// =====================================================================
#define FIR_TAPS 41
__constant__ float c_fir_coeffs[FIR_TAPS];


// =====================================================================
// KERNEL 1: TELEPHONY VOICE BANDPASS FILTER (IIR)
// =====================================================================
// Equations (1st-Order Cascaded Biquad):
//   High-Pass (300Hz):  $y_{hp}[n] = \alpha_{hp} \cdot (y_{hp}[n-1] + x[n] - x[n-1])$
//   Low-Pass (3400Hz):  $y_{lp}[n] = y_{lp}[n-1] + \alpha_{lp} \cdot (y_{hp}[n] - y_{lp}[n-1])$
//
// Description:
//   Assigns exactly 1 thread per microphone to run a sequential recursive filter. 
//   Isolates the 300Hz - 3400Hz human speech band, stripping out both low-frequency 
//   rumble and high-frequency room hiss so the cross-correlation locks onto vowels.
__global__ void voice_bandpass_kernel(const int32_t* input, float* filtered_out, int samples, int tensor_channels, 
                                      float* state_x, float* state_hp, float* state_lp, bool first_run) {
    int m = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (m < tensor_channels) {
        // If it's the very first frame, initialize to the raw DC offset to prevent a startup pop.
        float prev_x = first_run ? (float)input[0 * tensor_channels + m] : state_x[m];
        float prev_hp = first_run ? 0.0f : state_hp[m];
        float prev_lp = first_run ? 0.0f : state_lp[m];

        const float hp_alpha = 0.962f; 
        const float lp_alpha = 0.307f; 

        for (int i = 0; i < samples; ++i) {
            float raw = (float)input[i * tensor_channels + m];
            
            float hp_y = hp_alpha * (prev_hp + raw - prev_x);
            prev_x = raw;
            prev_hp = hp_y;

            float lp_y = prev_lp + lp_alpha * (hp_y - prev_lp);
            prev_lp = lp_y;

            filtered_out[i * tensor_channels + m] = lp_y;
        }
        
        // Save state for the next 10ms network frame
        state_x[m] = prev_x;
        state_hp[m] = prev_hp;
        state_lp[m] = prev_lp;
    }
}
// =====================================================================
// KERNEL 2: STREAMLINED CROSS-CORRELATION
// =====================================================================
// Equation: 
//   R(tau) = (1 / N) * SUM_{i=0}^{N-1} ( x_ref[i] * x_tgt[i + tau] )
//
// Description:
//   Computes the raw cross-covariance between the reference mic (0) 
//   and the target mics across a sliding window of time lags (tau).
__global__ void xcorr_kernel(const float* filtered_input, float* xcorr_out, int samples, int tensor_channels, int max_lag, int ref_mic) {
    int m = blockIdx.x; 
    int tau = threadIdx.x - max_lag; 
    int lag_range = 2 * max_lag + 1;
    
    if (m < tensor_channels) {
        double sum = 0.0;
        int count = 0;
        // note start at lag 2 since first 2 samples are muted (0.0)
        for (int i = max_lag + 2; i < samples - max_lag; i++) {
            float ref = filtered_input[i * tensor_channels + ref_mic];
            float tgt = filtered_input[(i + tau) * tensor_channels + m];
            
            sum += ref * tgt;
            count++;
        }
        // Accumulate across the 10 frames
        xcorr_out[m * lag_range + threadIdx.x] += (float)(sum / count);
    }
}

// =====================================================================
// KERNEL 3: SUB-SAMPLE PEAK FINDER
// =====================================================================
// Parabolic Interpolation Equation:
//   delta = 0.5 * (y_{-1} - y_{+1}) / (y_{-1} - 2*y_0 + y_{+1})
//   tau_fractional = tau_integer + delta
//
// Description:
//   Finds the discrete integer lag with the highest correlation, then
//   fits a parabola to the peak and its two neighbors to find the true
//   continuous peak at sub-sample precision.
__global__ void find_fractional_peaks_kernel(const float* xcorr, int* steer_lags, float* frac_lags, 
                                             float* peak_vals, int mics, int lag_range, int max_lag) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < mics) {
        float max_v = -1e20f;
        int best_t = 0;
        int peak_idx = 0;

        // 1. Find discrete maximum
        for (int t = 0; t < lag_range; ++t) {
            float val = xcorr[m * lag_range + t];
            if (val > max_v) {
                max_v = val;
                best_t = t - max_lag;
                peak_idx = t;
            }
        }

        steer_lags[m] = best_t; 
        float frac_lag = (float)best_t;
        peak_vals[m] = max_v;

        // 2. Parabolic Interpolation
        if (peak_idx > 0 && peak_idx < lag_range - 1) {
            float y1 = xcorr[m * lag_range + (peak_idx - 1)]; // y_{-1}
            float y2 = xcorr[m * lag_range + peak_idx];       // y_0
            float y3 = xcorr[m * lag_range + (peak_idx + 1)]; // y_{+1}

            float denom = (y1 - 2.0f * y2 + y3);
            if (fabsf(denom) > 1e-6f) {
                float delta = 0.5f * (y1 - y3) / denom;
                delta = fmaxf(-1.0f, fminf(1.0f, delta));
                frac_lag += delta;
            }
        }
        frac_lags[m] = frac_lag;
    }
}

// =====================================================================
// KERNEL 4: CONSTANT MEMORY STEERED BEAMFORMER
// =====================================================================
// Equations:
//   Delay-and-Sum: $y_{das}[t] = \frac{1}{M} \sum_{m=0}^{M-1} x_m[\text{clamp}(t + \tau_m, 0, N-1)]$
//   FIR Filter:    $y_{beam}[n] = \sum_{j=0}^{J-1} c_j \cdot y_{das}[\max(n-j, 0)]$
//
// Description:
//   A Delay-and-Sum beamformer cascaded into a Finite Impulse Response (FIR) filter.
//   The delay-and-sum logic is calculated dynamically for each historical 
//   time step required by the FIR filter. If the historical index goes below 0,
//   it is clamped to 0. Since the FIR is a zero-sum bandpass filter, applying 
//   it to a constant historical DC offset perfectly annihilates the transient 
//   click at chunk boundaries.
__global__ void steered_beamformer_kernel(
    const int32_t* input, 
    int32_t* output, 
    const int* steering_lags,
    int samples, int mics, int taps) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < samples) {
        float filtered = 0.0f;
        
        // FIR Convolution Loop using historical time steps
        for (int j = 0; j < taps; ++j) {
            int hist_idx = tid - j;
            
            // Boundary Clamp: Pretend the past was identical to the first sample.
            // This elegantly suppresses the massive DC offset transient!
            if (hist_idx < 0) {
                hist_idx = 0;
            }
            
            // Reconstruct the steered sum for this specific historical moment
            float hist_steered_sum = 0.0f;
            for (int m = 0; m < mics; ++m) {
                int lag = steering_lags[m];
                int shifted_idx = hist_idx + lag;
                
                // Spatial bounds clamp
                if (shifted_idx < 0) shifted_idx = 0;
                if (shifted_idx >= samples) shifted_idx = samples - 1;
                
                hist_steered_sum += (float)input[shifted_idx * mics + m];
            }
            hist_steered_sum /= (float)mics;
            
            // Apply the FIR tap to the historical sample
            filtered += hist_steered_sum * c_fir_coeffs[j];
        }
        output[tid] = (int32_t)filtered;
    }
}

void AudioBeamformerOp::compute(holoscan::InputContext& input, holoscan::OutputContext& op_output,
                              holoscan::ExecutionContext& context) {
    auto in_message = input.receive<holoscan::gxf::Entity>("input");
    if (!in_message) return;

    nvidia::gxf::Entity entity = static_cast<nvidia::gxf::Entity>(in_message.value());
    auto in_tensor_handle = entity.get<nvidia::gxf::Tensor>(""); 
    
    // 1. TENSOR CHANNELS: Used strictly for safe memory striding (will be 5 since last channel is beamformer output)
    int tensor_channels = num_channels_.get(); 
    int samples_per_ch = (in_tensor_handle.value()->size() / sizeof(int32_t)) / tensor_channels;
    
    // 2. PHYSICAL ARRAY: Used strictly for the 3D Math Geometry
    int physical_mics = ARRAY_PHYSICAL_MICS; 
    int ref_mic = ARRAY_REF_MIC;
    
    int max_lag = TRACKING_MAX_LAG;

    auto beam_tensor_handle = entity.add<nvidia::gxf::Tensor>("beam_output");
    const nvidia::gxf::Shape shape{samples_per_ch, 1};
    auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_.get()->gxf_cid());
    beam_tensor_handle.value()->reshape<int32_t>(shape, nvidia::gxf::MemoryStorageType::kDevice, allocator_handle.value());

    const int32_t* d_in = reinterpret_cast<const int32_t*>(in_tensor_handle.value()->pointer());
    int32_t* d_out = reinterpret_cast<int32_t*>(beam_tensor_handle.value()->pointer());
    
    int lag_range = 2 * max_lag + 1;

    if (!d_steering_lags_) {
        // Allocate all memory based on the incoming tensor size to prevent overflows
        cudaMalloc(&d_steering_lags_, tensor_channels * sizeof(int));
        cudaMemset(d_steering_lags_, 0, tensor_channels * sizeof(int));
        
        cudaMalloc(&d_fractional_lags_, tensor_channels * sizeof(float)); 
        cudaMalloc(&d_peak_vals_, tensor_channels * sizeof(float)); 
        cudaMalloc(&d_filtered_input_, samples_per_ch * tensor_channels * sizeof(float));  
        
        cudaMalloc(&d_correlation_results_, tensor_channels * lag_range * sizeof(float));
        cudaMemset(d_correlation_results_, 0, tensor_channels * lag_range * sizeof(float));
        
        // NEW: Allocate persistent IIR state
        cudaMalloc(&d_iir_x_, tensor_channels * sizeof(float));
        cudaMalloc(&d_iir_y_hp_, tensor_channels * sizeof(float));
        cudaMalloc(&d_iir_y_lp_, tensor_channels * sizeof(float));
    }

    if (!fir_initialized_) {
        cudaMemcpyToSymbol(c_fir_coeffs, d_fir_coeffs_, FIR_TAPS * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        fir_initialized_ = true;
    }

    int threads = 256;
    int blocks = (samples_per_ch + threads - 1) / threads;

    // Execute baseline kernels over all memory lanes to maintain stride integrity
    steered_beamformer_kernel<<<blocks, threads>>>(
        d_in, d_out, d_steering_lags_, samples_per_ch, tensor_channels, FIR_TAPS);

    voice_bandpass_kernel<<<1, tensor_channels>>>(d_in, d_filtered_input_, samples_per_ch, tensor_channels, 
                                                  d_iir_x_, d_iir_y_hp_, d_iir_y_lp_, first_run_);
    first_run_ = false;

    xcorr_kernel<<<tensor_channels, lag_range>>>(d_filtered_input_, d_correlation_results_, samples_per_ch, tensor_channels, max_lag, ref_mic);

    samples_processed_ += samples_per_ch;
    
    if (samples_processed_ >= (sample_rate_.get() / 10)) {
        
        find_fractional_peaks_kernel<<<1, tensor_channels>>>(d_correlation_results_, d_steering_lags_, d_fractional_lags_, d_peak_vals_, tensor_channels, lag_range, max_lag);

        std::vector<float> h_fractional_lags(tensor_channels);
        cudaMemcpy(h_fractional_lags.data(), d_fractional_lags_, tensor_channels * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> h_peak_vals(tensor_channels);
        cudaMemcpy(h_peak_vals.data(), d_peak_vals_, tensor_channels * sizeof(float), cudaMemcpyDeviceToHost);

        // =====================================================================
        // SQUELCH / VOICE ACTIVITY DETECTION
        // =====================================================================
        // Dynamically measure the actual reference mic
        float avg_power = h_peak_vals[ref_mic] / 10.0f; 
        float rms = sqrtf(fmaxf(avg_power, 1.0f));
        float dbfs = 20.0f * log10f(rms / 2147483648.0f); 
        float squelch_threshold = squelch_threshold_.get();
        
        last_dbfs_ = dbfs;
        float alpha = ema_alpha_.get(); 

        if (dbfs < squelch_threshold) {
            squelch_hold_frames_++;

            if (squelch_hold_frames_ > hang_frames_.get()) { 
                cudaMemset(d_steering_lags_, 0, tensor_channels * sizeof(int));
                // Return smoothly to center when the room goes quiet
                last_azimuth_   = last_azimuth_ + alpha * (0.0f - last_azimuth_);
                last_elevation_ = last_elevation_ + alpha * (0.0f - last_elevation_);
            }                         
        } else {
            squelch_hold_frames_ = 0;

            std::vector<int> h_lags(tensor_channels);
            for (int m = 0; m < tensor_channels; ++m) h_lags[m] = (int)roundf(h_fractional_lags[m]);
            cudaMemcpy(d_steering_lags_, h_lags.data(), tensor_channels * sizeof(int), cudaMemcpyHostToDevice);

            float fs = (float)sample_rate_.get();
            const float kSpeedOfSound = 34300.0f; 

            // PULL GEOMETRY FROM GLOBAL MACROS
            float mic_x[4] = ARRAY_MIC_X;
            float mic_y[4] = ARRAY_MIC_Y;

            float HTH[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
            float HTd[2] = {0.0f, 0.0f};

            // =====================================================================
            // 3D SPHERICAL TRIANGULATION (LEAST SQUARES REGRESSION)
            // =====================================================================
            // CRITICAL FIX: Loop STRICTLY terminates at `physical_mics` (4).
            // Prevents the matrix from reading out-of-bounds array memory and crashing to -90.0
            for (int m = 0; m < physical_mics; ++m) {
                if (m == ref_mic) continue; 

                float dt = h_fractional_lags[m] / fs;
                float expected_dist = kSpeedOfSound * dt;
                float dist = -expected_dist; 

                // Shift origin dynamically to the reference mic
                float dx = mic_x[m] - mic_x[ref_mic];
                float dy = mic_y[m] - mic_y[ref_mic];

                HTH[0][0] += dx * dx;
                HTH[0][1] += dx * dy;
                HTH[1][0] += dy * dx;
                HTH[1][1] += dy * dy;

                HTd[0] += dx * dist;
                HTd[1] += dy * dist;
            }

            float det = (HTH[0][0] * HTH[1][1]) - (HTH[0][1] * HTH[1][0]);
            float Sx = 0.0f;
            float Sy = 0.0f; 

            if (fabsf(det) > 1e-6f) {
                float inv_det = 1.0f / det;
                Sx = ( HTH[1][1] * HTd[0] - HTH[0][1] * HTd[1]) * inv_det;
                Sy = (-HTH[1][0] * HTd[0] + HTH[0][0] * HTd[1]) * inv_det;
            }

            Sx = fmaxf(-1.0f, fminf(1.0f, Sx));
            Sy = fmaxf(-1.0f, fminf(1.0f, Sy));

            // Right-Handed Spherical Projection
            /*float raw_az_deg = asinf(Sx) * 57.2958f; // (180/PI)
            float raw_el_deg = asinf(Sy) * 57.2958f; 
            */
// NEW: True 3D Spherical Projection
            // Recover the Z-axis (forward) vector using the Pythagorean identity
            float Sz_sq = 1.0f - (Sx * Sx) - (Sy * Sy);
            float Sz = sqrtf(fmaxf(0.0f, Sz_sq)); 

            // atan2f correctly calculates Azimuth independently of Elevation!
            float raw_az_deg = atan2f(Sx, Sz) * 57.2958f; 
            float raw_el_deg = asinf(Sy) * 57.2958f;

            // Exponential Moving Average
            last_azimuth_   = last_azimuth_   + alpha * (raw_az_deg - last_azimuth_);
            last_elevation_ = last_elevation_ + alpha * (raw_el_deg - last_elevation_);
        }
        
        if (verbose_.get()) {
           if (dbfs >= squelch_threshold) {
              HSB_LOG_INFO("BEAM | Level: {:.1f} dBFS | Az: {:.1f}° | El: {:.1f}°", 
                         dbfs, last_azimuth_, last_elevation_);
           } 
           else {
            HSB_LOG_INFO("BEAM | SQUELCHED | Level: {:.1f} dBFS < {:.1f} dBFS | Centering...", 
                         dbfs, squelch_threshold);
           }
        }
        
        samples_processed_ = 0;
        
        // CRITICAL FIX: Reset the accumulator for the next 100ms tracking window
        cudaMemset(d_correlation_results_, 0, tensor_channels * lag_range * sizeof(float));
    }

    auto track_tensor = entity.add<nvidia::gxf::Tensor>("tracking_info");
    track_tensor.value()->reshape<float>(nvidia::gxf::Shape{3}, nvidia::gxf::MemoryStorageType::kDevice, allocator_handle.value());
    
    float h_track[3] = {last_azimuth_, last_elevation_, last_dbfs_};
    cudaMemcpy(track_tensor.value()->pointer(), h_track, 3 * sizeof(float), cudaMemcpyHostToDevice);

    op_output.emit(in_message.value(), "output");
}
} // namespace hololink::operators
