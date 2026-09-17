# Analog Devices Sensor Enablement for NVIDIA Holoscan Sensor Bridge

This directory contains Analog Devices (ADI) sensor integrations, GPU-accelerated operators, Python bindings, ROS2 applications, and reference pipelines for the NVIDIA® Holoscan Sensor Bridge ecosystem.

The goal of these examples is to demonstrate how ADI sensing technologies can be seamlessly integrated into NVIDIA Holoscan workflows for real-time robotics, industrial automation, edge AI, autonomous systems, and multi-modal sensor fusion applications.

Current support includes:

- **3D Depth Sensing** using the **ADCAM3175-2M-EBZ** Time-of-Flight camera
- **Industrial IMUs** including **ADIS16505** and upcoming **ADIS16607**
- **Spatial Audio Processing** using **A2B® (AD2428/AD2427)** microphone systems
- **ROS2 Integration**
- **GPU-Accelerated Sensor Processing**
- **Holoviz Visualization**

---

# Supported Hardware

| Category | Part Number | Status | Description | Product |
|-----------|-------------|----------|-------------|-------------|
| 3D Depth Camera | **ADCAM3175-2M-EBZ** | ✅ Supported | Time-of-Flight (ToF) 3D depth camera | https://www.analog.com/en/products/adtf3175.html |
| Industrial IMU | **ADIS16505** | ✅ Supported | Precision industrial inertial measurement unit | https://www.analog.com/en/products/adis16505.html |
| Industrial IMU | **ADIS16607** | 🚧 Upcoming | Next-generation industrial inertial measurement unit | https://www.analog.com/en/products/adis16607.html |
| A2B Audio | **AD2428 + AD2427** | ✅ Supported | Multi-channel microphone acquisition and transport | https://www.analog.com/en/products/ad2428.html |
| Sensor Connectivity | Holoscan Sensor Bridge | ✅ Supported | FPGA-based sensor connectivity platform | Lattice & Microchip |
| AI Compute | NVIDIA Jetson / IGX / Thor | ✅ Supported | GPU accelerated AI processing platforms | Jetson Thor/AGX Orix/ IGX/ DGX Spark |

---

# Directory Structure

```text
examples/adi
│
├── aditof/                ADCAM3175-2M-EBZ 3D Depth Camera
│
├── adi_imu/               ADIS16505 / ADIS16607 IMU Support
│
├── a2baudio/              A2B Audio + Beamforming Pipeline
│
├── ros2_setup.sh          ROS2 Environment Setup
│
└── CMakeLists.txt
```

---

# Module 1 – ADCAM3175-2M-EBZ 3D Depth Camera

Directory:

```text
aditof/
```

## Overview

This module enables the **ADCAM3175-2M-EBZ** 3D depth camera within the NVIDIA Holoscan environment.

The implementation includes:

- Camera discovery and configuration
- Device control APIs
- Calibration support
- Frame capture
- CUDA-based frame unpacking
- Playback utilities
- Firmware management
- Sensor Bridge integration

The module provides the foundation for building:

- Spatial perception pipelines
- Robotics vision systems
- Obstacle detection
- Human-machine interaction
- Industrial inspection systems
- SLAM and mapping applications

## Key Components

| Component | Description |
|------------|------------|
| adcam_lib | Camera control library |
| adcam_unpack_op | GPU accelerated depth frame unpacking |
| adcam_player | Playback utility |
| adcam_calibration | Camera calibration support |
| adsd3500_flash | Firmware flashing support |
| programmer | Device programming utilities |

---

# Module 2 – Industrial IMU Integration

Directory:

```text
adi_imu/
```

## Supported Devices

| Device | Status |
|----------|--------|
| ADIS16505 | ✅ Supported |
| ADIS16607 | 🚧 Upcoming |

## Overview

This module integrates ADI industrial-grade inertial sensors into NVIDIA Holoscan.

The implementation supports:

- IMU acquisition
- Python bindings
- Holoscan operators
- ROS2 publishing
- Visualization workflows
- Timestamp synchronization

Measurements include:

- Accelerometer data
- Gyroscope data
- Temperature
- Sensor timestamps

## Architecture

```text
ADIS16505 / ADIS16607
          │
          ▼
    Sensor Bridge
          │
          ▼
      Holoscan
          │
    ┌─────┴─────┐
    ▼           ▼
Processing    ROS2
    │           │
    ▼           ▼
 Holoviz      RViz
```

## Key Components

| Component | Description |
|------------|------------|
| adi_imu_op | IMU acquisition operator |
| adi_imu_ros2.py | ROS2 publisher application |
| adi_imu_op_python | Python interface |
| adi_imu_visualization.launch.py | RViz visualization support |

## Example Use Cases

- Visual-Inertial Odometry (VIO)
- Robot localization
- Navigation
- Sensor fusion
- Mobility platforms
- Humanoid robots
- Autonomous Mobile Robots (AMRs)

---

# Module 3 – A2B Audio Processing Pipeline

Directory:

```text
a2baudio/
```

## Overview

This module demonstrates a complete A2B audio acquisition and processing pipeline using ADI Audio Bus technology.

Reference platform:

```text
AD2428 A2B Master
         │
         ▼
AD2427 Microphone Node
         │
         ▼
 4-Microphone Array
         │
         ▼
 Holoscan Sensor Bridge
         │
         ▼
 NVIDIA GPU
```

Captured audio streams are processed using GPU-accelerated operators for visualization, DSP, recording, and beamforming applications.

## Features

- Multi-channel microphone capture
- I2S audio reception
- Real-time waveform rendering
- FFT visualization
- Digital beamforming
- Audio recording
- CUDA acceleration
- Holoviz integration

## Key Components

### I2S Receiver

```text
i2s/
```

Receives audio streams from the Sensor Bridge.

### Beamformer

```text
audio_beamformer/
```

CUDA accelerated audio beamformer.

Features:

- Multi-channel processing
- Directional beam steering
- Real-time operation
- GPU acceleration

### Waveform Generator

```text
audio_waveform/
```

Generates waveforms for real-time visualization.

### Audio Recorder

```text
audio_filewriter/
```

Stores captured audio to disk.

### Visualization Application

```text
app/audio_viz.cpp
```

Demonstrates:

- Audio acquisition
- Real-time DSP
- FFT generation
- Beamforming
- Holoviz rendering

## Example Robotics Applications

- Voice-controlled robots
- Human-robot interaction
- Industrial acoustic monitoring
- Sound source localization
- Spatial audio systems
- Voice analytics

---

# ROS2 Integration

ROS2 support is provided for sensor visualization and interoperability.

Components include:

```text
ros2_setup.sh

adi_imu/
 ├── adi_imu_ros2.py
 └── adi_imu_visualization.launch.py
```

Typical capabilities:

- Sensor publishers
- ROS2 message generation
- RViz visualization
- Multi-sensor fusion workflows

---

# Build

Same as docker build as pubmished in Holoscan Sensor Bridge github

```bash
docker/build.sh --<igpu or dgpu>
```
or

From the root of the Holoscan Sensor Bridge workspace:

```bash
cmake -B build
cmake --build build -j$(nproc)
```

or

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

---

# Example Applications

## 3D Depth Camera

```bash
python adcam_player.py
```

## IMU + ROS2

```bash
python adi_imu_ros2.py
```

Visualization (run in a separate shell and not in the container shell):

```bash
ros2 launch adi_imu_visualization.launch.py
```

## A2B Audio Visualization

```bash
audio_viz_display
```

---

# Technology Stack

- NVIDIA Holoscan
- Holoscan Sensor Bridge
- CUDA
- C++
- Python
- ROS2
- Holoviz
- RViz
- A2B Audio
- Industrial IMUs
- 3D Time-of-Flight Sensing

---

# Multi-Modal Sensor Fusion

These examples demonstrate how multiple sensing modalities can be fused within a common Holoscan framework:

```text
3D Depth Camera
       │
       ▼
      GPU
       ▲
       │
Industrial IMU
       │
       ▼
   Holoscan
       ▲
       │
  A2B Audio
```

This enables development of advanced perception systems for:

- Autonomous Mobile Robots (AMRs)
- Humanoid Robotics
- Industrial Automation
- Spatial AI
- Edge AI
- Autonomous Systems

These examples illustrate how Analog Devices sensing technologies are enabled in NVIDIA Holoscan AI workloads by enabling:

- 3D Depth Perception
- Industrial Inertial Sensing
- Spatial Audio Processing
- GPU Accelerated DSP
- Real-Time Sensor Fusion

Together, ADI sensors and NVIDIA accelerated computing provide a scalable platform for next-generation robotics and intelligent edge systems.

---

# License

Refer to the main repository license and contributing guidelines for usage restrictions and contribution policies.

# Support
1. Reach out to FPGA vendor for getting access to RTL that supports ADCAM and other modules from Analog Devices
2. Email Holo.Scan@analog.com if you need to reach Analog Devices Holoscan support
