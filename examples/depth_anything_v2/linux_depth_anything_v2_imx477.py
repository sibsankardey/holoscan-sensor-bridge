# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Depth Anything V2 running on an IMX477 attached to a Holoscan Sensor Bridge,
# using the Linux socket receiver (no ConnectX required).
#
# This script is adapted from https://github.com/nvidia-holoscan/holohub/tree/main/applications/depth_anything_v2/
# repository.
#
# Source model for this instance is obtained from https://aihub.qualcomm.com/models/depth_anything_v2?searchTerm=depth+anything
#
# *NOTE: The user is responsible for checking if the model license is suitable for the intended purpose.*
#
# See README.md for detailed information.

"""
Download model (outside docker) inside the directory of: holoscan-sensor-bridge/

    wget https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/depth_anything_v2/releases/v0.59.0/depth_anything_v2-onnx-float.zip
    unzip depth_anything_v2-onnx-float.zip -d ./examples/depth_anything_v2

Start holoscan docker and install opencv(this step would need to be done every time the docker is restarted)
    pip install opencv-python-headless

Navigate to examples/depth_anything_v2/depth_anything_v2-onnx-float/ and run the following(inside docker demo):
    trtexec --onnx=depth_anything_v2.onnx   --saveEngine=depth.engine.fp32
    cp depth.engine.fp32 ../../../
    cd ../../../

Run the demo itself:
    python3 examples/depth_anything_v2/linux_depth_anything_v2_imx477.py


"""

import argparse
import ctypes
import logging
import os

import cuda.bindings.driver as cuda
import cupy as cp
import holoscan
import numpy as np
from depth import PostprocessorOp
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity

import hololink as hololink_module


class FormatInferenceInputOp(Operator):
    """Operator to format input image for inference"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed")).get()
        # OBS: Numpy conversion and moveaxis is needed to avoid strange
        # strides issue when doing inference
        tensor = np.moveaxis(tensor, 2, 0)[None]
        tensor = cp.asarray(tensor)

        # Create output message
        out_message = Entity(context)
        out_message.add(holoscan.as_tensor(tensor), "preprocessed")
        op_output.emit(out_message, "out")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        frame_limit,
        engine,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._frame_limit = frame_limit
        self._engine = engine

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        frame_context = self._cuda_context
        receiver_operator = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        bayer_format = self._camera.bayer_format()
        pixel_format = self._camera.pixel_format()
        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            optical_black=100,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgb_components_per_pixel = 3
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * rgb_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=False,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        image_shift = hololink_module.operators.ImageShiftToUint8Operator(
            self, name="image_shift", shift=8
        )

        #
        pool = holoscan.resources.UnboundedAllocator(self)
        preprocessor = holoscan.operators.FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **self.kwargs("preprocessor"),
        )
        format_input = FormatInferenceInputOp(
            self,
            name="transpose",
            pool=pool,
        )
        inference_args = self.kwargs("inference")
        inference_args["model_path_map"] = {"depth": self._engine}
        inference = holoscan.operators.InferenceOp(
            self,
            name="inference",
            allocator=pool,
            **inference_args,
        )
        postprocessor = PostprocessorOp(self, name="postprocessor", allocator=pool)

        # Register the mouse/framebuffer callbacks that drive PostprocessorOp's
        # display modes (original / depth / side-by-side / interactive).
        visualizer = holoscan.operators.HolovizOp(
            self,
            allocator=pool,
            name="holoviz",
            window_title="DepthAnything v2",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            mouse_button_callback=postprocessor.toggle_display_mode,
            cursor_pos_callback=postprocessor.cursor_pos_callback,
            framebuffer_size_callback=postprocessor.framebuffer_size_callback,
            **self.kwargs("holoviz"),
        )

        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, image_shift, {("transmitter", "input")})
        self.add_flow(image_shift, preprocessor, {("output", "")})
        self.add_flow(preprocessor, postprocessor, {("tensor", "input_image")})
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "input_depthmap")})
        self.add_flow(postprocessor, visualizer, {("output_image", "receivers")})
        self.add_flow(postprocessor, visualizer, {("output_specs", "input_specs")})

        # Not using metadata
        self.enable_metadata(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    default_configuration = os.path.join(os.path.dirname(__file__), "depth.yaml")
    parser.add_argument(
        "--configuration", default=default_configuration, help="Configuration file"
    )
    parser.add_argument(
        "--model-path",
        default="depth.engine.fp32",
        help="Depth Anything V2 model (or TRT engine; set inference.is_engine_path)",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        choices=(0, 1),
        help="which camera to stream: 0 to stream camera connected to j14 or 1 to stream camera connected to j17 (default is 0)",
    )
    parser.add_argument(
        "--resolution",
        default="4k",
        help="4k or 1080p",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=0x05,
        help="Configure exposure.",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    # Get a handle to the camera
    md = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(md, args.cam)
    hololink_channel = hololink_module.DataChannel(md)
    camera = hololink_module.sensors.imx477.Imx477(
        hololink_channel, args.cam, args.resolution
    )

    # Set up the application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        camera,
        args.frame_limit,
        args.model_path,
    )
    application.config(args.configuration)
    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()
    try:
        hololink.reset()
        camera.configure()

        # IMX477 Analog gain settings function. Analog gain value range is 0-1023 in decimal (10 bits). Users are free to experiment with the register values.
        camera.set_analog_gain(0x2FF)
        camera.set_exposure_reg(args.exposure)

        application.run()
    finally:
        hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
