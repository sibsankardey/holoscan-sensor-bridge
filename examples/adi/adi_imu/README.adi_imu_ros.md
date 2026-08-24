ADI IMU HOLOSCAN TO ROS 2 BRIDGE
This repository provides a high-performance, containerized integration between an
Analog Devices ADIS16505-2 and ADIS16607-2 Precision IMUs, the NVIDIA Holoscan Sensor Bridge
(HSB) 2.5.0 and 2.7.0, and ROS 2 (Jazzy).

It uses a custom C++ Holoscan operator to handle low-latency SPI transactions
and a Python native ROS 2 node to publish sensor_msgs/Imu
and sensor_msgs/Temperature data. It includes hardware-level UNIX epoch
timestamping to eliminate integration drift in downstream filters.

The example does NOT handle interrupts since the FPGA cannot do that now.  This means,
polling over the network link occurs.  From an improvement perspective, it is possible
for the FPGA to handle the DR pin interrupt and do the burst read similar to what is
in the code.

Prior to building, the Dockerfile.demo MUST be changed prior to container build:

The multi-stage container definition builds the C++ bindings, installs ROS 2,
configures CycloneDDS, and sets the critical PYTHONPATH and LD_LIBRARY_PATH
environment variables.  These env variables MUST be changed prior to building
container to match the user in the container.  For the CycloneDDS, the
<NetworkInterfaceAddress> must match the network interface name used.

FILE STRUCTURE
examples/adi/adi_imu/app/adi_imu_ros2.py
The primary execution script. Defines the Holoscan application graph,
instantiates the C++ hardware operator, and handles 32-bit payload parsing,
physical scaling, dynamic calibration, and ROS 2 publishing.

examples/adi/adi_imu/app/adi_imu_config.yaml
The hardware configuration file. Defines the network target, SPI parameters,
hardware pinouts, and physical scaling constants. (See "Configuration Details" below).

examples/adi/adi_imu/adi_imu_op.cpp
The low-level C++ hardware driver. Implements a polymorphic Strategy Pattern 
(Adis16505Driver and Adis16607Driver) to safely isolate Full-Duplex and 
Half-Duplex SPI transactions. It dynamically handles DPLL lock sequencing, 
CRC4 payload validation, and injects user calibration biases directly into 
IMU during initialization.

examples/adi/adi_imu/app/adi_imu_visualization.launch.py
A ROS 2 launch file designed to be run OUTSIDE the container. It spins up a
properly tuned Madgwick filter and an instance of RViz2 pre-configured with
Best Effort QoS and 3D camera controls.

YAML CONFIGURATION DETAILS & ADAPTING TO NEW IMUS
The adi_imu_config.yaml file controls how the Holoscan Sensor Bridge talks to
the physical IMU via the Lattice HSB J20 header. If you swap the ADIS16505-2
for a different ADI IMU, you must update this file according to the new sensor's datasheet.

Key Parameters:

hsb_ip: The IP address of the Holoscan Sensor Bridge (default 192.168.0.2).

sync_freq: External sync pulse frequency in Hz. Must match the external tool if sync_pin is used (default 120).

output_freq: The target polling and ROS 2 message output rate in Hz. Limited by internal IMU decimation math (default 240).

spi_port & spi_cs: The physical SPI bus and Chip Select line used on the HSB (default 2 and 0).

spi_cpol & spi_cpha: The SPI Clock Polarity and Phase. Most ADI IMUs use Mode 3
(CPOL=1, CPHA=1), but always verify in the datasheet.

spi_div: The SPI clock divider. This sets the SPICLK rate (default 10 - 1 MHz). 
         If a new IMU supports faster SPI clocks, or requires slower ones to prevent bit-dropping during
         burst reads, adjust this divider.
reset_pin: The GPIO pin mapped to the IMU's hardware reset line (default 9).

Data Ready (dr_pin)
The IMU uses its own internal oscillator to sample data only if sync_pin is -1.
When a sample is finished, the IMU pulses the DR pin. The Holoscan bridge polls 
this pin for a positive edge via SPI read. This provides the lowest noise but
means the IMU clock dictates the network frequency and is limited by the polling
rate - i.e. about 500 Hz. (Default: 8)

Sync Mode (sync_pin)
The Holoscan bridge (or external clock like a 1PPS signal) generates a pulse
and sends it to the IMU's SYNC pin via the J20 HSB header. This forces the IMU to 
utilize its own PLL and operate in sync mode with a limited rate up to 128 Hz on
sync pin and uses the decimation ratio in the IMU to support update IMU updates
based on the data ready pin.  SPI polling is still utilized for detection of the
positive edge. This is utilize where the IMU and Camera frames must be
captured at the exact same time or phase aligned to prevent visual-inertial drift. (Default: 13)
NOTE:  To utilize the sync pin properly, software from Lattice must be utilized
       to configure the sync pin.  What has been tested is the sync pin at 120 Hz.  This
       is a limitation on the ADIS16505-2 with max SYNC rate of 128 Hz.  The sync-out 
       source code must be obtained from Lattice.
       ./hololink-sync-out --mode=4 --delay-ns=0 --start-level=0 --exposure-ns=100000

local_gravity: The exact gravitational pull at your physical elevation and 
latitude in m/s^2. This is critical for preventing Z-axis integration drift 
during the automated calibration phase. (Default: 9.80665)

Physical Scaling (imu_models)
The YAML also stores hardware-agnostic scaling factors to convert raw 32-bit registers into ROS 2 standard units (rad/s and m/s^2).
ADIS16505-2 Defaults:
  gyro_scale: 0.000000006657734125
  accel_scale: 0.000000037384033203
ADIS16607-2 Defaults:
# 24-bit physical data is left-justified into a 32-bit integer in C++ (/256.0).
  # 16,000 LSB / (deg/sec)
  gyro_scale: 0.0000010908307825      
  
  # 200,000 LSB / g (Must be converted to m/s^2)
  accel_scale: 0.00004903325

ROS 2 MIDDLEWARE & NETWORKING (CYCLONEDDS)
Because high-frequency IMU data can easily overwhelm default DDS configurations,
this project relies strictly on CycloneDDS (rmw_cyclonedds_cpp) for network
transport.

The Docker container automatically configures CycloneDDS to handle large message
sizes and multicast traffic.  Note that ROS_DOMAIN_ID is set in the Dockerfile.

CRITICAL: Any external machines, tools, or host terminals interacting with this
data stream MUST also be configured to use CycloneDDS and matching domain IDs.
Mixing DDS vendors (e.g., FastRTPS with Cyclone) will result in dropped packets
or failure to discover the IMU topics.

NETWORKING REQUIREMENTS
By default, if the kernel is doing PREEMPT_RT (via uname -r), however must use busy network
polling via the following commands:
 sudo sysctl -w net.core.busy_read=50
 sudo sysctl -w net.core.busy_poll=50
Without these commands, the baseline will not go past 190ish Hz IMU message rate.

BUILD INSTRUCTIONS
CRITICAL HOLOLINK CORE PATCH:
To support the 30+ byte half-duplex burst payloads required by the ADIS16607-2, a core Hololink library file must be patched to bypass the default 16-byte DGX transaction limit.
File: src/hololink/core/hololink.cpp
Change: Around line 986 in the spi_transaction function, increase the write_command_count limit:
- if (write_command_count >= 16) {
+ if (write_command_count >= 40) {

This project must be built inside the Docker container to isolate dependencies.
Prior to building the container per Holoscan Sensor Bridge documentation, please
check the network interface to be used for ROS in the Cyclone DDS section.

Once inside the container:
cd holoscan-sensor-bridge
mkdir build;cd build
cmake ..
make -j$(nproc) adi_imu

Assumption is Lattice's sync output tool source code is available.  To build
the sync_out tool:
cd build
cmake --build . --target hololink-sync-out

USAGE & COMMAND LINE ARGUMENTS (INSIDE CONTAINER)
Prior to running the ROS2 node, the FPGA must be configured to generate the
sync pulse (120 Hz) to the IMU via - source available from Lattice:
./hololink-sync-out --mode=4 --delay-ns=0 --start-level=0 --exposure-ns=100000

Run the primary Python node from the root directory inside the container:
cd holoscan-sensor-bridge/examples
python3 adi_imu_ros2.py [Arguments]

Arguments:
--imu_model  : Select the target IMU hardware model. Options: ADIS16505-2 or ADIS16607-2 (default: ADIS16505-2)
--config     : Path to hardware configuration file (default: adi_imu_config.yaml)
--calibrate  : Run Calibration Mode for N seconds, output biases, and exit (float)

Bias Application Flags (Dynamically injected into IMU hardware DSP registers during initialization, maintaining zero-cost Python execution):
--calib_gx   : Gyroscope X-axis bias offset in rad/s (float)
--calib_gy   : Gyroscope Y-axis bias offset in rad/s (float)
--calib_gz   : Gyroscope Z-axis bias offset in rad/s (float)
--calib_ax   : Accelerometer X-axis bias offset in m/s^2 (float)
--calib_ay   : Accelerometer Y-axis bias offset in m/s^2 (float)
--calib_az   : Accelerometer Z-axis bias offset in m/s^2 (float)

HARDWARE CALIBRATION WORKFLOW
For precision applications, you must calibrate out resting biases. The built-in
calibration mode automatically detects and removes Earth's gravity vector (1G).

Keep the IMU perfectly still on a flat surface.

Run the calibration command for a specific duration (e.g., 60 seconds) based on
allan variance plots to set calibration duration.
python3 adi_imu_ros2.py --imu_model ADIS16607-2 --calibrate 60

Apply the results:
The script outputs Software Flags that you pass back into the Python script.
Crucially, these flags are now dynamically injected into the IMU's hardware RAM registers 
on boot, meaning the silicon does the math.

For the ADIS16505-2, it also outputs Hardware Register commands (echo strings). 
It is highly recommended to pipe the echo commands into the direct_reg_access 
interface to permanently burn the 32-bit biases to the silicon flash memory. 
(Note: Direct hardware flash burn commands for the ADIS16607-2 are currently 
bypassed in software pending final register verification).

VISUALIZATION (OUTSIDE CONTAINER)
It is highly recommended to run visualization tools natively on your HOST machine,
outside of the Docker container, to avoid complex X11 GUI forwarding issues.

Prerequisites (Host Machine):

ROS 2 Jazzy installed natively.

The imu_filter_madgwick and ros-jazzy-rviz-imu-plugin ROS 2 package installed.

RMW_IMPLEMENTATION set to rmw_cyclonedds_cpp.

ROS_DOMAIN_ID set to match the container (Default: 5).

To view the IMU data in 3D space, open a terminal on your host machine, source
your ROS 2 environment, and run the provided launch file. This starts a low-gain
Madgwick filter (must be installed) and RViz2:

ros2 launch adi_imu_visualization.launch.py
