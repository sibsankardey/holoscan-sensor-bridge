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
#include <chrono>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>

#include "../audio_beamformer/audio_beamformer_op.hpp"
#include "../audio_filewriter/audio_file_writer_op.hpp"
#include "../audio_waveform/audio_waveform_op.hpp"
#include <hololink/operators/linux_receiver/linux_receiver_op.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

// =====================================================================
// SYSTEM CONFIGURATION & TUNABLES
// =====================================================================
#define AUDIO_CHANNELS 4
#define AUDIO_SAMPLE_RATE 48000
#define AUDIO_NYQUIST (AUDIO_SAMPLE_RATE / 2.0f)

// =====================================================================
// LEGACY I2S CAPTURE BLOCK (CPNX100 register map for HSB 2.7.0)
// =====================================================================
namespace legacy_i2s {
constexpr uint32_t I2S_CTRL = 0x8000'0000;
constexpr uint32_t REG_CAPTURE_EN = 0x00;
constexpr uint32_t REG_CHUNK_SIZE = 0x04;
constexpr uint32_t REG_I2S_CFG = 0x08;
constexpr uint32_t REG_ID = 0x0C;
constexpr uint32_t ID_VALUE = 0x1234'BEEF;
constexpr uint32_t DEFAULT_CFG = 0x0004'2020;
constexpr uint8_t MAX_CHANNELS = 7;
} // namespace legacy_i2s

class LegacyI2s {
public:
    LegacyI2s(std::shared_ptr<hololink::Hololink> hololink, uint32_t base = legacy_i2s::I2S_CTRL)
        : hololink_(std::move(hololink))
        , base_(base)
    {
    }

    uint32_t read_id() { return hololink_->read_uint32(base_ + legacy_i2s::REG_ID); }

    void set_chunk_size(uint32_t chunk_size)
    {
        write(legacy_i2s::REG_CHUNK_SIZE, chunk_size);
    }

    void configure(uint32_t cfg_value)
    {
        write(legacy_i2s::REG_I2S_CFG, cfg_value);
    }

    void start_capture() { write(legacy_i2s::REG_CAPTURE_EN, 1); }
    void stop_capture() { write(legacy_i2s::REG_CAPTURE_EN, 0); }

private:
    void write(uint32_t offset, uint32_t value)
    {
        if (!hololink_->write_uint32(base_ + offset, value)) {
            throw std::runtime_error(
                fmt::format("ACK failure writing I2S register {:#x}.", base_ + offset));
        }
    }
    std::shared_ptr<hololink::Hololink> hololink_;
    const uint32_t base_;
};

// --- Tracking & DSP Constants ---
#define TRACKING_FOV_DEGREES 180.0f // Total FOV (+/- 90 degrees)
#define TRACKING_EMA_ALPHA 0.35f // Smoothing factor (0.0-heavy smoothing to 1.0-no smoothing)
#define TRACKING_HANG_FRAMES 20 // 10 frames = 1.0 seconds at 10Hz

namespace {

// =====================================================================
// OPERATOR: TRACKING TO SCREEN
// =====================================================================
class TrackingToScreenOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(TrackingToScreenOp)

    TrackingToScreenOp() = default;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input<holoscan::gxf::Entity>("input");
        spec.output<holoscan::gxf::Entity>("output");
        spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor memory.");
        spec.param(squelch_threshold_, "squelch_threshold", "Squelch", "dBFS limit", -65.0f);
        spec.param(fov_degrees_, "fov_degrees", "FOV", "Field of view", TRACKING_FOV_DEGREES);
    }

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) override
    {
        auto in_message = op_input.receive<holoscan::gxf::Entity>("input");
        if (!in_message)
            return;

        nvidia::gxf::Entity entity = static_cast<nvidia::gxf::Entity>(in_message.value());
        auto track_tensor_handle = entity.get<nvidia::gxf::Tensor>("tracking_info");

        if (track_tensor_handle) {
            float h_track[3];
            cudaMemcpy(h_track, track_tensor_handle.value()->pointer(), 3 * sizeof(float), cudaMemcpyDeviceToHost);

            float azimuth = h_track[0];
            float elevation = h_track[1];
            float dbfs = h_track[2];

            float fov = fov_degrees_.get();
            float half_fov = fov / 2.0f;

            // Convert Spherical Degrees to Normalized Screen Coordinates [0.0, 1.0]
            float screen_x = (azimuth + half_fov) / fov;
            float screen_y = (-elevation + half_fov) / fov;

            // Squelch Hide: Push off-screen if signal is too weak
            /* removed for now. if (dbfs < squelch_threshold_.get()) {
                  screen_x = -1.0f;
                  screen_y = -1.0f;
              } */

            float circle_radius = 0.03f;

            auto out_message = holoscan::gxf::Entity::New(&context);
            nvidia::gxf::Entity out_entity = static_cast<nvidia::gxf::Entity>(out_message);

            auto circle_tensor = out_entity.add<nvidia::gxf::Tensor>("circle_coords");
            auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_.get()->gxf_cid());

            circle_tensor.value()->reshape<float>(nvidia::gxf::Shape { 1, 3 }, nvidia::gxf::MemoryStorageType::kDevice, allocator_handle.value());

            float h_coords[3] = { screen_x, screen_y, circle_radius };
            cudaMemcpy(circle_tensor.value()->pointer(), h_coords, 3 * sizeof(float), cudaMemcpyHostToDevice);

            op_output.emit(out_message, "output");
        }
    }

private:
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<float> squelch_threshold_;
    holoscan::Parameter<float> fov_degrees_;
};

// =====================================================================
// APPLICATION GRAPH
// =====================================================================
class HoloscanApplication : public holoscan::Application {
public:
    explicit HoloscanApplication(bool headless, bool fullscreen, CUcontext cuda_context,
        hololink::DataChannel& hololink_channel, int64_t frame_limit, float gain,
        std::shared_ptr<LegacyI2s> i2s, size_t frame_size, size_t chunk_size,
        float min_freq, float max_freq, bool verbose,
        std::string record_file, float record_start, float record_stop,
        float squelch_dbfs)
        : headless_(headless)
        , fullscreen_(fullscreen)
        , cuda_context_(cuda_context)
        , hololink_channel_(hololink_channel)
        , frame_limit_(frame_limit)
        , gain_(gain)
        , i2s_(std::move(i2s))
        , frame_size_(frame_size)
        , chunk_size_(chunk_size)
        , min_freq_(min_freq)
        , max_freq_(max_freq)
        , verbose_(verbose)
        , record_file_(record_file)
        , record_start_(record_start)
        , record_stop_(record_stop)
        , squelch_dbfs_(squelch_dbfs)
    {
    }

    void compose() override
    {
        using namespace holoscan;

        std::shared_ptr<Condition> condition;
        if (frame_limit_) {
            condition = make_condition<CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<BooleanCondition>("ok", true);
        }

        auto i2s = i2s_;
        auto i2s_receiver = make_operator<hololink::operators::LinuxReceiverOp>(
            "i2s_receiver", condition,
            Arg("frame_size", frame_size_),
            Arg("frame_context", cuda_context_),
            Arg("hololink_channel", &hololink_channel_),
            Arg("device_start", std::function<void()>([i2s] {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                i2s->start_capture();
            })),
            Arg("device_stop", std::function<void()>([i2s] {
                i2s->stop_capture();
            })),
            Arg("trim", true));

        auto allocator = make_resource<holoscan::UnboundedAllocator>("allocator");

        auto beamformer = make_operator<hololink::operators::AudioBeamformerOp>(
            "beamformer",
            Arg("allocator", allocator),
            Arg("num_channels", AUDIO_CHANNELS),
            Arg("sample_rate", AUDIO_SAMPLE_RATE),
            Arg("squelch_threshold", squelch_dbfs_),
            Arg("ema_alpha", TRACKING_EMA_ALPHA),
            Arg("hang_frames", TRACKING_HANG_FRAMES),
            Arg("verbose", verbose_));

        auto tracking_to_screen = make_operator<TrackingToScreenOp>(
            "tracking_to_screen",
            Arg("allocator", allocator),
            Arg("squelch_threshold", squelch_dbfs_),
            Arg("fov_degrees", TRACKING_FOV_DEGREES));

        auto waveform_gen = make_operator<hololink::operators::AudioWaveformOp>(
            "waveform_gen",
            Arg("allocator", allocator),
            Arg("num_channels", AUDIO_CHANNELS + 1),
            Arg("gain", gain_),
            Arg("apply_window", true),
            Arg("min_freq", min_freq_),
            Arg("max_freq", max_freq_),
            Arg("sample_rate", AUDIO_SAMPLE_RATE),
            Arg("verbose", verbose_));

        auto audio_writer = make_operator<hololink::operators::AudioFileWriterOp>(
            "audio_writer",
            Arg("file_path", record_file_),
            Arg("sample_rate", AUDIO_SAMPLE_RATE),
            Arg("record_start", record_start_),
            Arg("record_stop", record_stop_));

        std::vector<holoscan::ops::HolovizOp::InputSpec> visualizer_inputs;

        holoscan::ops::HolovizOp::InputSpec grid_spec { "grid_lines", holoscan::ops::HolovizOp::InputType::LINES };
        grid_spec.color_ = { 0.2f, 0.2f, 0.2f, 1.0f };
        visualizer_inputs.push_back(grid_spec);

        std::vector<std::vector<float>> colors = {
            { 1.0f, 0.2f, 0.2f, 1.0f }, // Ch 0: Red
            { 0.2f, 1.0f, 0.2f, 1.0f }, // Ch 1: Green
            { 0.2f, 0.8f, 1.0f, 1.0f }, // Ch 2: Cyan
            { 1.0f, 0.8f, 0.2f, 1.0f } // Ch 3: Yellow
        };

        for (int i = 0; i < AUDIO_CHANNELS; ++i) {
            holoscan::ops::HolovizOp::InputSpec spec { fmt::format("channel_{}", i), holoscan::ops::HolovizOp::InputType::LINE_STRIP };
            spec.color_ = colors[i % 4];
            visualizer_inputs.push_back(spec);
        }

        holoscan::ops::HolovizOp::InputSpec beam_spec { fmt::format("channel_{}", AUDIO_CHANNELS), holoscan::ops::HolovizOp::InputType::LINE_STRIP };
        beam_spec.color_ = { 1.0f, 1.0f, 1.0f, 1.0f };
        visualizer_inputs.push_back(beam_spec);

        holoscan::ops::HolovizOp::InputSpec circle_spec { "circle_coords", holoscan::ops::HolovizOp::InputType::OVALS };
        circle_spec.color_ = { 0.0f, 1.0f, 0.0f, 1.0f };
        circle_spec.line_width_ = 4;
        visualizer_inputs.push_back(circle_spec);

        auto visualizer = make_operator<holoscan::ops::HolovizOp>("holoviz",
            Arg("fullscreen", fullscreen_),
            Arg("headless", headless_),
            Arg("tensors", visualizer_inputs));
        visualizer->metadata_policy(holoscan::MetadataPolicy::kUpdate);

        // --- DAG Routing ---
        add_flow(i2s_receiver, beamformer, { { "output", "input" } });

        add_flow(beamformer, waveform_gen, { { "output", "input" } });
        add_flow(waveform_gen, visualizer, { { "output", "receivers" } });

        add_flow(beamformer, tracking_to_screen, { { "output", "input" } });
        add_flow(tracking_to_screen, visualizer, { { "output", "receivers" } });

        add_flow(beamformer, audio_writer, { { "output", "input" } });
    }

private:
    const bool headless_;
    const bool fullscreen_;
    const CUcontext cuda_context_;
    hololink::DataChannel& hololink_channel_;
    const int64_t frame_limit_;
    const float gain_;
    std::shared_ptr<LegacyI2s> i2s_;
    const size_t frame_size_;
    const size_t chunk_size_;
    const float min_freq_;
    const float max_freq_;
    const bool verbose_;
    const std::string record_file_;
    const float record_start_;
    const float record_stop_;
    const float squelch_dbfs_;
};

} // anonymous namespace

int main(int argc, char** argv)
{
    bool headless = false;
    bool fullscreen = false;
    bool verbose = false;
    int64_t frame_limit = 0;
    float gain = 1.0f;
    float min_freq = 0.0f;
    float max_freq = AUDIO_NYQUIST;
    float squelch_dbfs = -65.0f;
    std::string hololink_ip = "192.168.0.2";
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;

    std::string record_file = "test.wav";
    float record_start = 0.0f;
    float record_stop = 0.0f;

    uint32_t i2s_address = 0x80000000;
    int sensor_number = 2;
    double frame_duration_ms = 10.0;

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 },
        { "verbose", no_argument, nullptr, 'v' },
        { "frame-limit", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "gain", required_argument, nullptr, 0 },
        { "i2s-address", required_argument, nullptr, 0 },
        { "sensor", required_argument, nullptr, 'S' },
        { "frame-duration-ms", required_argument, nullptr, 'f' },
        { "freq-min", required_argument, nullptr, 0 },
        { "freq-max", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { "record-file", required_argument, nullptr, 0 },
        { "record-start", required_argument, nullptr, 0 },
        { "record-stop", required_argument, nullptr, 0 },
        { "squelch-dbfs", required_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 }
    };

    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "hS:f:g:v", long_options, &option_index);
        if (c == -1)
            break;

        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const struct option* cur_option = &long_options[option_index];
            if (cur_option->name == std::string("headless"))
                headless = true;
            else if (cur_option->name == std::string("fullscreen"))
                fullscreen = true;
            else if (cur_option->name == std::string("frame-limit"))
                frame_limit = std::stoll(argument);
            else if (cur_option->name == std::string("hololink"))
                hololink_ip = argument;
            else if (cur_option->name == std::string("gain"))
                gain = std::stof(argument);
            else if (cur_option->name == std::string("i2s-address"))
                i2s_address = std::stoul(argument, nullptr, 16);
            else if (cur_option->name == std::string("sensor"))
                sensor_number = std::stoi(argument, nullptr, 0);
            else if (cur_option->name == std::string("frame-duration-ms"))
                frame_duration_ms = std::stod(argument, nullptr);
            else if (cur_option->name == std::string("freq-min"))
                min_freq = std::stof(argument);
            else if (cur_option->name == std::string("freq-max"))
                max_freq = std::stof(argument);
            else if (cur_option->name == std::string("record-file"))
                record_file = argument;
            else if (cur_option->name == std::string("record-start"))
                record_start = std::stof(argument);
            else if (cur_option->name == std::string("record-stop"))
                record_stop = std::stof(argument);
            else if (cur_option->name == std::string("squelch-dbfs"))
                squelch_dbfs = std::stof(argument);
        } else if (c == 'h') {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -h, --help              Display this help message\n"
                      << "  -v, --verbose           Enable verbose logging\n"
                      << "  -S, --sensor NUM        Sensor number (default 2)\n"
                      << "  -f, --frame-duration-ms MS  Frame duration in ms (default 10.0)\n"
                      << "  -g, --gain FLOAT        Audio gain (default 1.0)\n"
                      << "  --headless              Run without display\n"
                      << "  --fullscreen            Run in fullscreen mode\n"
                      << "  --hololink IP           Hololink IP address (default 192.168.0.2)\n"
                      << "  --i2s-address HEX       I2S register base address (default 0x80000000)\n"
                      << "  --frame-limit N         Stop after N frames (0 = unlimited)\n"
                      << "  --freq-min HZ           Minimum display frequency (default 0)\n"
                      << "  --freq-max HZ           Maximum display frequency (default Nyquist)\n"
                      << "  --record-file PATH      WAV output file (default test.wav)\n"
                      << "  --record-start SEC      Start recording at time (default 0)\n"
                      << "  --record-stop SEC       Stop recording at time (default 0 = disabled)\n"
                      << "  --squelch-dbfs DB       Squelch threshold in dBFS (default -65)\n"
                      << "  --log-level LEVEL       Log level\n";
            return EXIT_SUCCESS;
        } else if (c == 'v') {
            verbose = true;
        }
    }

    constexpr size_t BYTES_PER_SAMPLE = 4;
    size_t chunk_size = static_cast<size_t>(AUDIO_SAMPLE_RATE * BYTES_PER_SAMPLE * (frame_duration_ms / 1000.0));
    size_t frame_size = chunk_size * 4;

    try {
        holoscan::set_log_level(log_level);
        CudaCheck(cuInit(0));
        CUdevice cu_device;
        CudaCheck(cuDeviceGet(&cu_device, 0));
        CUcontext cu_context;
        CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

        hololink::Metadata channel_metadata = hololink::Enumerator::find_channel(hololink_ip);
        hololink::DataChannel::use_sensor(channel_metadata, sensor_number);
        hololink::DataChannel hololink_channel(channel_metadata);

        std::shared_ptr<hololink::Hololink> hololink = hololink_channel.hololink();
        hololink->start();
        // commenting out reset to avoid other application using hololink to stop working
        // hololink->reset();

        auto i2s = std::make_shared<LegacyI2s>(hololink, i2s_address);

        uint32_t id = i2s->read_id();
        std::cout << "I2S ID register: 0x" << std::hex << std::setw(8) << std::setfill('0')
                  << id << std::dec << std::endl;
        if (id != legacy_i2s::ID_VALUE) {
            std::cerr << "WARNING: I2S ID mismatch (expected 0x1234BEEF). "
                         "The loaded bitstream may not support this register map."
                      << std::endl;
        }

        i2s->set_chunk_size(chunk_size);
        i2s->configure(legacy_i2s::DEFAULT_CFG);

        auto application = holoscan::make_application<HoloscanApplication>(
            headless, fullscreen, cu_context, hololink_channel, frame_limit, gain,
            i2s, frame_size, chunk_size, min_freq, max_freq, verbose,
            record_file, record_start, record_stop, squelch_dbfs);

        application->run();

        i2s->stop_capture();
        hololink->stop();
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
