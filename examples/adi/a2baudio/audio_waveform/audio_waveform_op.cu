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
#include "audio_waveform_op.hpp"
#include <gxf/std/tensor.hpp>
#include <gxf/std/allocator.hpp>
#include <fmt/format.h>
#include <hololink/core/logging_internal.hpp>
#include <math_constants.h>
#include <vector>

namespace hololink::operators {

// =====================================================================
// KERNEL 1: SIGNAL CONDITIONING (PRE-EMPHASIS & WINDOWING)
// =====================================================================
// UPDATED: Now accepts 'stride' and 'offset' to dynamically handle both
// interleaved 4-channel audio AND flat 1-channel beamformer audio.
__global__ void apply_window_kernel(const int32_t* audio_in, cufftReal* fft_in, 
                                    int samples, int stride, int offset, bool apply_window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < samples) {
        // Safely extract the memory based on whether it is raw or beam
        int32_t raw_val = audio_in[(idx * stride) + offset];
        float float_val = (float)raw_val;

        if (idx > 0) {
            int32_t prev_val = audio_in[((idx - 1) * stride) + offset];
            float_val = float_val - 0.95f * (float)prev_val;
        }

        float normalized = float_val / 2147483648.0f; 
        
        if (apply_window) {
            float window = 0.5f * (1.0f - cosf(2.0f * CUDART_PI_F * idx / (samples - 1)));
            fft_in[idx] = normalized * window;
        } else {
            fft_in[idx] = normalized;
        }
    }
}

// =====================================================================
// KERNEL 2: PARALLEL POWER ACCUMULATOR
// =====================================================================
__global__ void accumulate_power_kernel(const cufftComplex* fft_out, double* accumulated_power, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bins) {
        double re = (double)fft_out[idx].x;
        double im = (double)fft_out[idx].y;
        accumulated_power[idx] += (re * re + im * im);
    }
}

// =====================================================================
// KERNEL 3: TREE REDUCTION FOR RMS & PEAK FINDING
// =====================================================================
__global__ void reduce_stats_kernel(const double* acc_power, int num_bins, int min_bin, int max_bin,
                                    double* out_sum, double* out_max_val, int* out_max_idx) {
    __shared__ double s_sum[512];
    __shared__ double s_max[512];
    __shared__ int s_max_idx[512];

    int tid = threadIdx.x;
    double my_sum = 0.0;
    double my_max = -1.0;
    int my_max_idx = -1;

    for (int i = tid; i < num_bins; i += blockDim.x) {
        double val = acc_power[i];
        my_sum += val;
        
        if (i >= min_bin && i <= max_bin && val > my_max) {
            my_max = val;
            my_max_idx = i;
        }
    }

    s_sum[tid] = my_sum;
    s_max[tid] = my_max;
    s_max_idx[tid] = my_max_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            if (s_max[tid + s] > s_max[tid]) {
                s_max[tid] = s_max[tid + s];
                s_max_idx[tid] = s_max_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_sum[0] = s_sum[0];
        out_max_val[0] = s_max[0];
        out_max_idx[0] = s_max_idx[0];
    }
}

// =====================================================================
// KERNEL 4: VISUAL MAGNITUDE CALCULATION
// =====================================================================
__global__ void compute_magnitude_kernel(const cufftComplex* fft_out, float2* channel_out, 
                                         int plot_bins, int bin_min, int num_channels, int current_channel, 
                                         float gain, int samples, float hz_per_bin, float min_freq, float max_freq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < plot_bins) {
        int fft_idx = bin_min + idx; 
        
        float freq = fft_idx * hz_per_bin;
        float min_f = fmaxf(min_freq, 1.0f); 
        float max_f = fmaxf(max_freq, min_f + 1.0f); 
        
        float log_min = log10f(min_f);
        float log_max = log10f(max_f);
        
        float x;
        if (freq <= min_f) x = 0.0f;
        else if (freq >= max_f) x = 1.0f;
        else x = (log10f(freq) - log_min) / (log_max - log_min);
        
        float mag = sqrtf(fft_out[fft_idx].x * fft_out[fft_idx].x + fft_out[fft_idx].y * fft_out[fft_idx].y);
        mag /= (float)samples;
        if (fft_idx > 0) mag *= 2.0f; 
        
        float mag_clamped = fmaxf(mag, 1e-7f); 
        float db = 20.0f * log10f(mag_clamped);
        db += gain;
        
        float min_db = -100.0f;
        float max_db = 0.0f;
        float normalized_y = (db - min_db) / (max_db - min_db);
        normalized_y = fmaxf(0.0f, fminf(normalized_y, 1.0f));
        
        float baseline_y = (float)(current_channel + 1) / (float)num_channels;
        float max_height = 0.95f / (float)num_channels;
        
        float y = baseline_y - (normalized_y * max_height);
        
        channel_out[idx] = make_float2(x, y);
    }
}

void generate_grid_tensors(nvidia::gxf::Entity& out_message, nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator,
                           float min_freq, float max_freq) {
    std::vector<float2> host_lines;
    float min_f = fmaxf(min_freq, 1.0f);
    float max_f = fmaxf(max_freq, min_f + 1.0f);
    float log_min = log10f(min_f);
    float log_max = log10f(max_f);

    for (float power = 1.0f; power <= 4.0f; power += 1.0f) {
        float base_freq = powf(10.0f, power); 
        for (int mult = 1; mult < 10; ++mult) {
            float freq = base_freq * mult;
            if (freq >= min_f && freq <= max_f) {
                float x = (log10f(freq) - log_min) / (log_max - log_min);
                host_lines.push_back(make_float2(x, 0.0f)); 
                host_lines.push_back(make_float2(x, 1.0f)); 
            }
        }
    }

    if (!host_lines.empty()) {
        auto out_tensor = out_message.add<nvidia::gxf::Tensor>("grid_lines");
        out_tensor.value()->reshape<float>(
            nvidia::gxf::Shape{static_cast<int>(host_lines.size()), 2}, 
            nvidia::gxf::MemoryStorageType::kDevice, allocator
        );
        cudaMemcpy(out_tensor.value()->pointer(), host_lines.data(), 
                   host_lines.size() * sizeof(float2), cudaMemcpyHostToDevice);
    }
}

void AudioWaveformOp::setup(holoscan::OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(allocator_, "allocator", "Allocator", "Allocator for the output tensor");
    spec.param(num_channels_, "num_channels", "Number of Channels", "E.g., 4 for TDM4", 4);
    spec.param(gain_, "gain", "Gain", "Visual amplitude multiplier", 1.0f);
    spec.param(apply_window_, "apply_window", "Apply Window", "Apply Hann window before FFT", true);
    spec.param(min_freq_, "min_freq", "Min Frequency", "Lower bound for FFT plot in Hz", 0.0f);
    spec.param(max_freq_, "max_freq", "Max Frequency", "Upper bound for FFT plot in Hz", 24000.0f);
    spec.param(sample_rate_, "sample_rate", "Sample Rate", "Audio sample rate in Hz", 48000);
    spec.param(verbose_, "verbose", "Verbose Logging", "Enable console printouts", false);
}

void AudioWaveformOp::compute(holoscan::InputContext& input, holoscan::OutputContext& op_output,
                              holoscan::ExecutionContext& context) {
    auto in_message = input.receive<holoscan::gxf::Entity>("input");
    if (!in_message) return;

    // 1. EXTRACT BOTH TENSORS
    auto raw_tensor = in_message.value().get<holoscan::Tensor>("");
    auto beam_tensor = in_message.value().get<holoscan::Tensor>("beam_output");
    
    if (!raw_tensor || !beam_tensor) throw std::runtime_error("Missing audio tensors");

    // audio_viz.cpp asks for 5 channels (4 raw + 1 beam)
    int total_channels = num_channels_.get(); 
    int raw_channels = total_channels - 1; // 4
    
    float gain = gain_.get();
    bool apply_window = apply_window_.get();
    int sample_rate = sample_rate_.get(); 
    
    size_t data_bytes = raw_tensor->nbytes();
    int total_raw_samples = data_bytes / sizeof(int32_t); 
    int samples_per_channel = total_raw_samples / raw_channels;
    if (samples_per_channel == 0) return;

    samples_processed_ += samples_per_channel;
    frames_processed_++;
    bool print_snr = (samples_processed_ >= sample_rate);

    int num_bins = samples_per_channel / 2 + 1;

    // Allocate persistent GPU buffers for FFT and Math Reductions
    if (samples_per_channel != allocated_samples_) {
        if (d_fft_in_) cudaFree(d_fft_in_);
        if (d_fft_out_) cudaFree(d_fft_out_);
        if (fft_plan_) cufftDestroy(fft_plan_);
        if (d_accumulated_power_) cudaFree(d_accumulated_power_);
        if (d_out_sum_) cudaFree(d_out_sum_);
        if (d_out_max_val_) cudaFree(d_out_max_val_);
        if (d_out_max_idx_) cudaFree(d_out_max_idx_);

        cudaMalloc(&d_fft_in_, samples_per_channel * sizeof(cufftReal));
        cudaMalloc(&d_fft_out_, num_bins * sizeof(cufftComplex));
        cufftPlan1d(&fft_plan_, samples_per_channel, CUFFT_R2C, 1);
        
        cudaMalloc(&d_accumulated_power_, total_channels * num_bins * sizeof(double));
        cudaMemset(d_accumulated_power_, 0, total_channels * num_bins * sizeof(double));

        cudaMalloc(&d_out_sum_, sizeof(double));
        cudaMalloc(&d_out_max_val_, sizeof(double));
        cudaMalloc(&d_out_max_idx_, sizeof(int));

        allocated_samples_ = samples_per_channel;
    }

    float nyquist = sample_rate / 2.0f;
    float hz_per_bin = nyquist / (float)(num_bins - 1); 

    int bin_min = std::max(0, std::min(num_bins - 1, static_cast<int>(min_freq_.get() / hz_per_bin)));
    int bin_max = std::max(0, std::min(num_bins - 1, static_cast<int>(max_freq_.get() / hz_per_bin)));
    if (bin_max < bin_min) bin_max = bin_min; 
    int plot_bins = bin_max - bin_min + 1;

    auto out_message = nvidia::gxf::Entity::New(context.context());
    auto allocator_handle_expected = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_.get()->gxf_cid());
    if (!allocator_handle_expected) throw std::runtime_error("Failed to get Allocator");
    auto allocator_handle = allocator_handle_expected.value();

    const int32_t* d_raw_in = reinterpret_cast<const int32_t*>(raw_tensor->data());
    const int32_t* d_beam_in = reinterpret_cast<const int32_t*>(beam_tensor->data());

    int threads = 256;
    int blocks_in = (samples_per_channel + threads - 1) / threads;
    int blocks_bins = (num_bins + threads - 1) / threads;
    int blocks_out = (plot_bins + threads - 1) / threads;

    generate_grid_tensors(out_message.value(), allocator_handle, min_freq_.get(), max_freq_.get());

    const nvidia::gxf::Shape shape{plot_bins, 2}; 
    
    // Process ALL 5 channels dynamically
    for (int ch = 0; ch < total_channels; ++ch) {
        
        // DYNAMIC STRIDING: 
        // If ch < 4: Read from d_raw_in using a 4-stride
        // If ch == 4: Read from d_beam_in using a 1-stride
        const int32_t* d_active_in = (ch < raw_channels) ? d_raw_in : d_beam_in;
        int active_stride = (ch < raw_channels) ? raw_channels : 1;
        int active_offset = (ch < raw_channels) ? ch : 0;
        
        apply_window_kernel<<<blocks_in, threads>>>(d_active_in, d_fft_in_, samples_per_channel, active_stride, active_offset, apply_window);
        cufftExecR2C(fft_plan_, d_fft_in_, d_fft_out_);

        double* channel_acc_ptr = d_accumulated_power_ + (ch * num_bins);
        accumulate_power_kernel<<<blocks_bins, threads>>>(d_fft_out_, channel_acc_ptr, num_bins);

        if (print_snr) {
            int min_search_bin = std::max(1, static_cast<int>(500.0f / hz_per_bin));
            int max_search_bin = std::min(num_bins - 1, static_cast<int>(4500.0f / hz_per_bin));

            reduce_stats_kernel<<<1, 512>>>(channel_acc_ptr, num_bins, min_search_bin, max_search_bin,
                                            d_out_sum_, d_out_max_val_, d_out_max_idx_);

            double h_sum, h_max_val;
            int h_max_idx;
            cudaMemcpy(&h_sum, d_out_sum_, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_max_val, d_out_max_val_, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_max_idx, d_out_max_idx_, sizeof(int), cudaMemcpyDeviceToHost);

            double p_sum = h_sum / frames_processed_;
            double mean_square = p_sum / ((double)samples_per_channel * samples_per_channel);
            if (apply_window) mean_square *= 2.666666667; 
            
            double rms = std::sqrt(mean_square);
            double dbfs = 20.0 * std::log10(rms + 1e-12); 
            float peak_freq = h_max_idx * hz_per_bin;
            
            if (verbose_.get()) {
                if (ch < raw_channels) {
                    HSB_LOG_INFO("CH {}: Sustained Peak = {:.0f} Hz | Level = {:.2f} dBFS", ch, peak_freq, dbfs);
                } else {
                    HSB_LOG_INFO("BEAM : Sustained Peak = {:.0f} Hz | Level = {:.2f} dBFS", peak_freq, dbfs);
                }
            }
        }

        std::string tensor_name = fmt::format("channel_{}", ch);
        auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(tensor_name.c_str());
        out_tensor.value()->reshape<float>(shape, nvidia::gxf::MemoryStorageType::kDevice, allocator_handle);
        float2* d_channel_out = reinterpret_cast<float2*>(out_tensor.value()->pointer());

        compute_magnitude_kernel<<<blocks_out, threads>>>(
            d_fft_out_, d_channel_out, plot_bins, bin_min, total_channels, ch, gain, samples_per_channel,
            hz_per_bin, min_freq_.get(), max_freq_.get() 
        );
    }
    
    if (print_snr) {
        cudaMemset(d_accumulated_power_, 0, total_channels * num_bins * sizeof(double));
        samples_processed_ = 0; 
        frames_processed_ = 0;
    }
    
    op_output.emit(out_message.value(), "output");
}

void AudioWaveformOp::stop() {
    if (d_fft_in_) cudaFree(d_fft_in_);
    if (d_fft_out_) cudaFree(d_fft_out_);
    if (fft_plan_) cufftDestroy(fft_plan_);
    if (d_accumulated_power_) cudaFree(d_accumulated_power_);
    if (d_out_sum_) cudaFree(d_out_sum_);
    if (d_out_max_val_) cudaFree(d_out_max_val_);
    if (d_out_max_idx_) cudaFree(d_out_max_idx_);
}

} // namespace hololink::operators
