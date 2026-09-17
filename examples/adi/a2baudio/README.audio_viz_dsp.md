# Acoustic Tracking & Beamforming Pipeline
A real-time, GPU-accelerated digital signal processing (DSP) pipeline built on the Nvidia Holoscan SDK for the Jetson Orin AGX
platform with Holoscan Sensor Bridge. This application ingests 4-channel raw I2S microphone data, performs Time Difference of
Arrival (TDOA) spatial tracking, and executes Steered Delay-and-Sum (DAS) beamforming. 

The output is simultaneously visualized as a continuous multi-channel spectrogram of each microphore and at the bottom the
beamformer audio with a 3D tracking overlay and written to disk as a 16-bit PCM `.wav` file.

This code has been tested in Holoscan Sensor Bridge 2.7.0

## Architecture & Data Flow
This application utilizes a Directed Acyclic Graph (DAG) to maximize parallel execution on the GPU. The heavy spatial math is
calculated once, and the resulting lightweight tensors are broadcast to parallel branches for rendering, analysis, and disk I/O.

### Holoscan Operators
1. **`I2sReceiverOp`**: Ingests raw 24-bit I2S audio (padded to 32-bit MSB-aligned integers) at 48 kHz directly from the Hololink board.
2. **`AudioBeamformerOp`**: Applies a cascaded IIR telephony filter, executes cross-correlation for TDOA tracking, calculates 3D spatial coordinates, and isolates the target voice using a Steered Delay-and-Sum beamformer cascaded into a 41-tap FIR bandpass filter.
3. **`TrackingToScreenOp`**: Translates spherical tracking coordinates (Azimuth, Elevation) into normalized screen coordinates and applies an acoustic squelch threshold.
4. **`AudioWaveformOp`**: Executes hardware-accelerated `cuFFT` for frequency domain conversion and utilizes a 512-thread GPU tree reduction to calculate 1Hz global peak and RMS power.
5. **`AudioFileWriterOp`**: Handles stream-safe disk I/O, writing the isolated beam to a `.wav` file (see Data Format section below).
6. **`HolovizOp`**: Native Holoscan operator for rendering the real-time, hardware-accelerated Vulkan user interface.

## Recording Audio & Saved Data Format
To save the isolated acoustic beam for downstream Machine Learning, Vision-Language-Action (VLA), or Vision-Language Model (VLM) analysis, use the recording flags:
```bash
audio_viz --record-file isolated_speech.wav --record-start 0.0 --record-stop 10.0

Command Line Arguments:
--help, -h
    Type: Flag
    Description: Prints the help menu and exits.
    Default: N/A

--headless
    Type: Flag
    Description: Runs the application completely without the Vulkan Holoviz UI.
    Default: false

--fullscreen
    Type: Flag
    Description: Runs the Vulkan Holoviz UI in fullscreen mode.
    Default: false

--verbose, -v
    Type: Flag
    Description: Enables 1Hz SNR, power, and beam-tracking console printouts.
    Default: false

--frame-limit
    Type: Integer
    Description: Stops execution after processing this many frames (0 = infinite).
    Default: 0

--hololink
    Type: String
    Description: IP address of the Hololink hardware board.
    Default: "192.168.0.2"

--gain, -g
    Type: Float
    Description: Visual amplitude multiplier for the spectrogram rendering.
    Default: 1.0

--i2s-address
    Type: Hex String
    Description: Memory address for the hardware I2S IP block (dependent on Lattice FPGA)
    Default: 0x80000000

--sensor, -S
    Type: Integer
    Description: Sensor number designation on the Hololink board.
    Default: 2

--frame-duration-ms, -f
    Type: Double
    Description: Hardware capture chunk size in milliseconds. (max is 20)
    Default: 10.0

--freq-min
    Type: Float
    Description: Lower frequency bound for the Holoviz FFT plot (Hz).
    Default: 0.0

--freq-max
    Type: Float
    Description: Upper frequency bound for the Holoviz FFT plot (Hz).
    Default: 24000.0 (Nyquist)
    
--squelch-dbfs
    Type: Float
    Description:  Threshold level of audio squelching - used for VAD.  Anything less than this
    level is considered silence and no beamformer activity will occur.  This should at least
    match the ambient noise floor.  To see what that is use the -v option.
    Default:  -65.0 dBFS 

--log-level
    Type: String
    Description: Holoscan logger verbosity (TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL).
    Default: INFO

--record-file
    Type: String
    Description: Target filename for the raw, un-gained 16-bit PCM output.
    Default: "test.wav"

--record-start
    Type: Float
    Description: Time (in seconds) to start writing the audio file.
    Default: 0.0

--record-stop
    Type: Float
    Description: Time (in seconds) to automatically stop writing the file (0.0 = infinite).
    Default: 0.0

To build within the holoscan-sensor-bridge directory(version 2.5.0) and on Orin AGX (not available for Thor due to Holoviz)
mkdir build; cd build
cmake ..
make -j$(nproc) audio_viz_disp

Executable will be in build/examples directory.

Parameters available for tweaking (i.e. calibration/tuning)
In examples/adi/a2baudio/app/audio_viz.cpp
 #define TRACKING_FOV_DEGREES 90.0f    <- based on the limits on the Holoviz edges (i.e. +/- 45 default)
 #define TRACKING_EMA_ALPHA 0.3f       <- Smoothing factor (0.0 to 1.0) - used by moving average filter, lower = slower response
 #define TRACKING_HANG_FRAMES 10       <- 10 frames = 1.0 seconds at 10Hz, # of frames allowed for voice activity logic,
 										  holds allows for x frames of silence for tracking voice.

In examples/adi/a2baudio/audio_beamformer/audio_beamformer_op.hpp
 int max_lag_ = 8  <- maximum lag allowed in xcorr based on speed of sound and max distance between mics in array

In examples/adi/a2baudio/audio_beamformer/audio_beamformer_op.cu, x/y coordinates for each mic in cms relative to mic0.
However the actual math requires the normal to the array and NOT looking at the array.
 The values used are based on a circular array (4.25 cm radius) with Mic 0 in the center but looking at the array
 here are the values:

                 [Mic 1] (0.0, 4.25)
                    / \
                   /   \
         4.25 cm  /     \  4.25 cm
                 /       \
                / [Mic 0] \
               / (0.0, 0.0)\
              /      |      \
             /    4.25 cm    \
            /        |        \
[Mic 3]----/---------|---------\----[Mic 2]
(-3.68, -2.125)               (3.68, -2.125)

Since all math is relative to the normal, flipping X coordinate sign is required.
 Perspective: "Looking out" from the array into the room.
 Channel Order due to FPGA: [Slot 1 (mic3), Slot 2 (mic2), Slot 3 (mic1), Slot 4 (mic0/Center/Reference)]
   float mic_x[4] = {3.68f, -3.68f, 0.0f, 0.0f};
   float mic_y[4] = {-2.125f, -2.125f, 4.25f, 0.0f};

 Thus for angles, the array must be oriented as above with Mic0 on the top.
