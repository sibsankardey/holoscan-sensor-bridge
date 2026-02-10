# Running Analog Devices Inc Time of Flight Chip ADTF3175 capture apps
More details about the chip and eval kit can be found on https://www.analog.com/en/products/adtf3175.html
The new module which is a varient of ADTF3175 includes a ToF sensor + 2xADSD3500 DSPs to do depth calculation 
inside DSP and produce MIPI 22-pin output. The MIPI output then fed to Microchip Polarfire HSB or equivalent HSB
supporting ADI ADSD3500 output. Please get in touch with Microchip to access to the bitfile/FW to support 
ADTF3175. The FW version used >=2507

# Prerequisites for running the application

Here is how to capture using the app:

Power on the module.

```bash

  python3 ./examples/aditof/adcam_player.py --resetAdcam 1

```

Capture depth data

```bash

  python3 ./examples/aditof/adcam_player.py --capture 1

```

This will open a Holoviz display and show 3 images (Depth, Absolute Brightness and Confidence image)

## Getting the source

```bash

  git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git

```

## Building the Hololink source and starting the Hololink container

```bash

  sh ./docker/build.sh

  export DISPLAY=`(DISPLAY)`

  xhost +

  sh ./docker/demo.sh

```


## Using locally built holoscan sdk image as base image

1. Build latest Holoscan SDK
1. Modify ./docker/Dockerfile to use latest SDK image as a base image. Image name would
   be holoscan-sdk-build-<arch>-<dgpu or igpu>
1. Build the source and restart container using the steps mentioned above.
1. Also mount SDK install directory and set PYTHONPATH to
   <holoscan-sdk>/public/install-<arch>-<dgpu or igpu>/python/lib/ inside the container.
1. Run the apps following above mentioned steps.
