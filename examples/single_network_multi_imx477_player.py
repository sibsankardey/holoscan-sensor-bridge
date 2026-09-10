# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Multi-camera variant of linux_single_network_stereo_imx477_player.py.
# Streams up to 4 IMX477 sensors over the Linux socket path in even intervals

import argparse
import ctypes
import logging

import cuda.bindings.driver as cuda
import holoscan

import hololink as hololink_module

_GRID_OFFSETS = [
    # (offset_x, offset_y, width, height)
    [(0.0, 0.0, 1.0, 1.0)],
    [
        (0.0, 0.0, 1.0, 0.5),  # cam 0: top
        (0.0, 0.5, 1.0, 0.5),  # cam 1: bottom
    ],
    [
        (0.0, 0.0, 0.5, 0.5),  # cam 0: top-left
        (0.5, 0.0, 0.5, 0.5),  # cam 1: top-right
        (0.0, 0.5, 0.5, 0.5),  # cam 2: bottom-left
    ],
    [
        (0.0, 0.0, 0.5, 0.5),  # cam 0: top-left
        (0.5, 0.0, 0.5, 0.5),  # cam 1: top-right
        (0.0, 0.5, 0.5, 0.5),  # cam 2: bottom-left
        (0.5, 0.5, 0.5, 0.5),  # cam 3: bottom-right
    ],
]


class MicroApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        ibv_name,
        ibv_port,
        hololink_channels,
        cameras,
        frame_limit,
        window_height,
        window_width,
        num_cameras,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._hololink_channels = hololink_channels
        self._cameras = cameras
        self._frame_limit = frame_limit
        self._window_height = window_height
        self._window_width = window_width
        self._num_cameras = num_cameras
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        logging.info("compose")
        conditions = []
        if self._frame_limit:
            for i in range(self._num_cameras):
                cond = holoscan.conditions.CountCondition(
                    self, name=f"count_{i}", count=self._frame_limit
                )
                conditions.append(cond)
        else:
            for i in range(self._num_cameras):
                cond = holoscan.conditions.BooleanCondition(
                    self, name=f"ok_{i}", enable_tick=True
                )
                conditions.append(cond)

        # All cameras share resolution, so size the pool from camera 0.
        cam0 = self._cameras[0]
        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=cam0._width * ctypes.sizeof(ctypes.c_uint16) * cam0._height,
            num_blocks=8,  # * self._num_cameras,
        )

        tensor_names = [f"cam{i}" for i in range(self._num_cameras)]
        csi_to_bayer_ops = []
        for i, camera in enumerate(self._cameras):
            op = hololink_module.operators.CsiToBayerOp(
                self,
                name=f"csi_to_bayer_{i}",
                allocator=csi_to_bayer_pool,
                cuda_device_ordinal=self._cuda_device_ordinal,
                out_tensor_name=tensor_names[i],
            )
            camera.configure_converter(op)
            csi_to_bayer_ops.append(op)

        frame_size = csi_to_bayer_ops[0].get_csi_length()
        frame_context = self._cuda_context

        receiver_ops = []
        for i, (channel, camera) in enumerate(
            zip(self._hololink_channels, self._cameras)
        ):
            op = hololink_module.operators.RoceReceiverOp(
                self,
                conditions[i],
                name=f"receiver_{i}",
                frame_size=frame_size,
                frame_context=frame_context,
                ibv_name=self._ibv_name,
                ibv_port=self._ibv_port,
                hololink_channel=channel,
                device=camera,
            )
            receiver_ops.append(op)

        pixel_format = cam0.pixel_format()
        bayer_format = cam0.bayer_format()

        image_processor_ops = []
        for i in range(self._num_cameras):
            op = hololink_module.operators.ImageProcessorOp(
                self,
                name=f"image_processor_{i}",
                optical_black=100,
                bayer_format=bayer_format.value,
                pixel_format=pixel_format.value,
            )
            image_processor_ops.append(op)

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=cam0._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * cam0._height,
            num_blocks=8,
        )

        demosaic_ops = []
        for i in range(self._num_cameras):
            op = holoscan.operators.BayerDemosaicOp(
                self,
                name=f"demosaic_{i}",
                pool=bayer_pool,
                generate_alpha=True,
                alpha_value=65535,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
                in_tensor_name=tensor_names[i],
                out_tensor_name=tensor_names[i],
            )
            demosaic_ops.append(op)

        viz_specs = []
        for i in range(self._num_cameras):
            spec = holoscan.operators.HolovizOp.InputSpec(
                tensor_names[i], holoscan.operators.HolovizOp.InputType.COLOR
            )
            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x, view.offset_y, view.width, view.height = _GRID_OFFSETS[
                self._num_cameras - 1
            ][i]
            spec.views = [view]
            viz_specs.append(spec)

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=viz_specs,
            height=self._window_height,
            width=self._window_width,
        )

        for i in range(self._num_cameras):
            self.add_flow(receiver_ops[i], csi_to_bayer_ops[i], {("output", "input")})
            self.add_flow(
                csi_to_bayer_ops[i], image_processor_ops[i], {("output", "input")}
            )
            self.add_flow(
                image_processor_ops[i], demosaic_ops[i], {("output", "receiver")}
            )
            self.add_flow(demosaic_ops[i], visualizer, {("transmitter", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    infiniband_devices = hololink_module.infiniband_devices()
    parser.add_argument(
        "--ibv-name",
        default=infiniband_devices[0],
        help="IBV device to use",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device",
    )
    parser.add_argument(
        "--pattern",
        action="store_true",
        help="Configure to display a test pattern.",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=0x05,
        help="Configure exposure.",
    )
    parser.add_argument(
        "--resolution",
        default="1080p",
        help="4k, 1080p, or 720p",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1080,  # arbitrary default
        help="Set the height of the displayed window",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1920,  # arbitrary default
        help="Set the width of the displayed window",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=2,
        choices=range(1, 5),
        help="Number of cameras to showcase (up to 4 if using Rev 2 board)",
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
    # New scope: Only allow 4 cameras if Rev2 board, else ignore the below
    # Check rev2 board based on uuid, if uuid then allow the next set, if not continue as normal
    # -------------------------------------------------------------------------
    uuid_strategy = hololink_module.BasicEnumerationStrategy(
        total_sensors=4,
        total_dataplanes=2,
        sifs_per_sensor=1,
    )
    rev2_uuid = "48484948-5045-4848-4853-564845484850"
    hololink_module.Enumerator.set_uuid_strategy(rev2_uuid, uuid_strategy)

    # -------------------------------------------------------------------------
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)

    hololink_channels = []
    cameras = []
    adjusted_resolution = args.resolution

    # Force resolution adjustment for multiple cameras
    if args.num_cameras > 2:
        adjusted_resolution = "720p"
    elif args.num_cameras == 2 and args.resolution == "4k":
        adjusted_resolution = "1080p"

    for i in range(args.num_cameras):
        meta = hololink_module.Metadata(channel_metadata)
        hololink_module.DataChannel.use_sensor(meta, i)
        channel = hololink_module.DataChannel(meta)
        camera = hololink_module.sensors.imx477.Imx477(
            channel, camera_id=i, resolution=adjusted_resolution
        )
        hololink_channels.append(channel)
        cameras.append(camera)

    # Set up the application
    application = MicroApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        args.ibv_name,
        args.ibv_port,
        hololink_channels,
        cameras,
        args.frame_limit,
        args.window_height * min(args.num_cameras, 2),
        args.window_width,
        args.num_cameras,
    )

    # Run it.
    hololink = hololink_channels[0].hololink()
    hololink.start()
    try:
        hololink.reset()
        # Configures each camera
        for camera in cameras:
            camera.configure()
            # Analog gain range is 0-1023 (10 bits); tweak as needed.
            camera.set_analog_gain(0x2FF)
            camera.set_exposure_reg(args.exposure)
            if args.pattern:
                camera.set_pattern()
        application.run()
    finally:
        hololink.stop()


if __name__ == "__main__":
    main()
