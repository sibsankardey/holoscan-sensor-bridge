#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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

# More details about the chip and eval kit can be found on https://www.analog.com/en/products/adtf3175.html

import argparse
import ctypes
import logging

import adcam
import cuda.bindings.driver as cuda
import cupy as cp
import holoscan

import hololink as hololink_module


class ADTFUnpackOp(holoscan.core.Operator):

    JET_LUT_U8 = bytes(
        [
            0x00,
            0x00,
            0x7F,
            0x00,
            0x00,
            0x84,
            0x00,
            0x00,
            0x88,
            0x00,
            0x00,
            0x8D,
            0x00,
            0x00,
            0x91,
            0x00,
            0x00,
            0x96,
            0x00,
            0x00,
            0x9A,
            0x00,
            0x00,
            0x9F,
            0x00,
            0x00,
            0xA3,
            0x00,
            0x00,
            0xA8,
            0x00,
            0x00,
            0xAC,
            0x00,
            0x00,
            0xB1,
            0x00,
            0x00,
            0xB6,
            0x00,
            0x00,
            0xBA,
            0x00,
            0x00,
            0xBF,
            0x00,
            0x00,
            0xC3,
            0x00,
            0x00,
            0xC8,
            0x00,
            0x00,
            0xCC,
            0x00,
            0x00,
            0xD1,
            0x00,
            0x00,
            0xD5,
            0x00,
            0x00,
            0xDA,
            0x00,
            0x00,
            0xDE,
            0x00,
            0x00,
            0xE3,
            0x00,
            0x00,
            0xE8,
            0x00,
            0x00,
            0xEC,
            0x00,
            0x00,
            0xF1,
            0x00,
            0x00,
            0xF5,
            0x00,
            0x00,
            0xFA,
            0x00,
            0x00,
            0xFE,
            0x00,
            0x00,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0x00,
            0x04,
            0xFF,
            0x00,
            0x08,
            0xFF,
            0x00,
            0x0C,
            0xFF,
            0x00,
            0x10,
            0xFF,
            0x00,
            0x14,
            0xFF,
            0x00,
            0x18,
            0xFF,
            0x00,
            0x1C,
            0xFF,
            0x00,
            0x20,
            0xFF,
            0x00,
            0x24,
            0xFF,
            0x00,
            0x28,
            0xFF,
            0x00,
            0x2C,
            0xFF,
            0x00,
            0x30,
            0xFF,
            0x00,
            0x34,
            0xFF,
            0x00,
            0x38,
            0xFF,
            0x00,
            0x3C,
            0xFF,
            0x00,
            0x40,
            0xFF,
            0x00,
            0x44,
            0xFF,
            0x00,
            0x48,
            0xFF,
            0x00,
            0x4C,
            0xFF,
            0x00,
            0x50,
            0xFF,
            0x00,
            0x54,
            0xFF,
            0x00,
            0x58,
            0xFF,
            0x00,
            0x5C,
            0xFF,
            0x00,
            0x60,
            0xFF,
            0x00,
            0x64,
            0xFF,
            0x00,
            0x68,
            0xFF,
            0x00,
            0x6C,
            0xFF,
            0x00,
            0x70,
            0xFF,
            0x00,
            0x74,
            0xFF,
            0x00,
            0x78,
            0xFF,
            0x00,
            0x7C,
            0xFF,
            0x00,
            0x80,
            0xFF,
            0x00,
            0x84,
            0xFF,
            0x00,
            0x88,
            0xFF,
            0x00,
            0x8C,
            0xFF,
            0x00,
            0x90,
            0xFF,
            0x00,
            0x94,
            0xFF,
            0x00,
            0x98,
            0xFF,
            0x00,
            0x9C,
            0xFF,
            0x00,
            0xA0,
            0xFF,
            0x00,
            0xA4,
            0xFF,
            0x00,
            0xA8,
            0xFF,
            0x00,
            0xAC,
            0xFF,
            0x00,
            0xB0,
            0xFF,
            0x00,
            0xB4,
            0xFF,
            0x00,
            0xB8,
            0xFF,
            0x00,
            0xBC,
            0xFF,
            0x00,
            0xC0,
            0xFF,
            0x00,
            0xC4,
            0xFF,
            0x00,
            0xC8,
            0xFF,
            0x00,
            0xCC,
            0xFF,
            0x00,
            0xD0,
            0xFF,
            0x00,
            0xD4,
            0xFF,
            0x00,
            0xD8,
            0xFF,
            0x00,
            0xDC,
            0xFE,
            0x00,
            0xE0,
            0xFA,
            0x00,
            0xE4,
            0xF7,
            0x02,
            0xE8,
            0xF4,
            0x05,
            0xEC,
            0xF1,
            0x08,
            0xF0,
            0xED,
            0x0C,
            0xF4,
            0xEA,
            0x0F,
            0xF8,
            0xE7,
            0x12,
            0xFC,
            0xE4,
            0x15,
            0xFF,
            0xE1,
            0x18,
            0xFF,
            0xDD,
            0x1C,
            0xFF,
            0xDA,
            0x1F,
            0xFF,
            0xD7,
            0x22,
            0xFF,
            0xD4,
            0x25,
            0xFF,
            0xD0,
            0x29,
            0xFF,
            0xCD,
            0x2C,
            0xFF,
            0xCA,
            0x2F,
            0xFF,
            0xC7,
            0x32,
            0xFF,
            0xC3,
            0x36,
            0xFF,
            0xC0,
            0x39,
            0xFF,
            0xBD,
            0x3C,
            0xFF,
            0xBA,
            0x3F,
            0xFF,
            0xB7,
            0x42,
            0xFF,
            0xB3,
            0x46,
            0xFF,
            0xB0,
            0x49,
            0xFF,
            0xAD,
            0x4C,
            0xFF,
            0xAA,
            0x4F,
            0xFF,
            0xA6,
            0x53,
            0xFF,
            0xA3,
            0x56,
            0xFF,
            0xA0,
            0x59,
            0xFF,
            0x9D,
            0x5C,
            0xFF,
            0x9A,
            0x5F,
            0xFF,
            0x96,
            0x63,
            0xFF,
            0x93,
            0x66,
            0xFF,
            0x90,
            0x69,
            0xFF,
            0x8D,
            0x6C,
            0xFF,
            0x89,
            0x70,
            0xFF,
            0x86,
            0x73,
            0xFF,
            0x83,
            0x76,
            0xFF,
            0x80,
            0x79,
            0xFF,
            0x7D,
            0x7C,
            0xFF,
            0x79,
            0x80,
            0xFF,
            0x76,
            0x83,
            0xFF,
            0x73,
            0x86,
            0xFF,
            0x70,
            0x89,
            0xFF,
            0x6C,
            0x8D,
            0xFF,
            0x69,
            0x90,
            0xFF,
            0x66,
            0x93,
            0xFF,
            0x63,
            0x96,
            0xFF,
            0x5F,
            0x9A,
            0xFF,
            0x5C,
            0x9D,
            0xFF,
            0x59,
            0xA0,
            0xFF,
            0x56,
            0xA3,
            0xFF,
            0x53,
            0xA6,
            0xFF,
            0x4F,
            0xAA,
            0xFF,
            0x4C,
            0xAD,
            0xFF,
            0x49,
            0xB0,
            0xFF,
            0x46,
            0xB3,
            0xFF,
            0x42,
            0xB7,
            0xFF,
            0x3F,
            0xBA,
            0xFF,
            0x3C,
            0xBD,
            0xFF,
            0x39,
            0xC0,
            0xFF,
            0x36,
            0xC3,
            0xFF,
            0x32,
            0xC7,
            0xFF,
            0x2F,
            0xCA,
            0xFF,
            0x2C,
            0xCD,
            0xFF,
            0x29,
            0xD0,
            0xFF,
            0x25,
            0xD4,
            0xFF,
            0x22,
            0xD7,
            0xFF,
            0x1F,
            0xDA,
            0xFF,
            0x1C,
            0xDD,
            0xFF,
            0x18,
            0xE0,
            0xFF,
            0x15,
            0xE4,
            0xFF,
            0x12,
            0xE7,
            0xFF,
            0x0F,
            0xEA,
            0xFF,
            0x0C,
            0xED,
            0xFF,
            0x08,
            0xF1,
            0xFC,
            0x05,
            0xF4,
            0xF8,
            0x02,
            0xF7,
            0xF4,
            0x00,
            0xFA,
            0xF0,
            0x00,
            0xFE,
            0xED,
            0x00,
            0xFF,
            0xE9,
            0x00,
            0xFF,
            0xE5,
            0x00,
            0xFF,
            0xE2,
            0x00,
            0xFF,
            0xDE,
            0x00,
            0xFF,
            0xDA,
            0x00,
            0xFF,
            0xD7,
            0x00,
            0xFF,
            0xD3,
            0x00,
            0xFF,
            0xCF,
            0x00,
            0xFF,
            0xCB,
            0x00,
            0xFF,
            0xC8,
            0x00,
            0xFF,
            0xC4,
            0x00,
            0xFF,
            0xC0,
            0x00,
            0xFF,
            0xBD,
            0x00,
            0xFF,
            0xB9,
            0x00,
            0xFF,
            0xB5,
            0x00,
            0xFF,
            0xB1,
            0x00,
            0xFF,
            0xAE,
            0x00,
            0xFF,
            0xAA,
            0x00,
            0xFF,
            0xA6,
            0x00,
            0xFF,
            0xA3,
            0x00,
            0xFF,
            0x9F,
            0x00,
            0xFF,
            0x9B,
            0x00,
            0xFF,
            0x98,
            0x00,
            0xFF,
            0x94,
            0x00,
            0xFF,
            0x90,
            0x00,
            0xFF,
            0x8C,
            0x00,
            0xFF,
            0x89,
            0x00,
            0xFF,
            0x85,
            0x00,
            0xFF,
            0x81,
            0x00,
            0xFF,
            0x7E,
            0x00,
            0xFF,
            0x7A,
            0x00,
            0xFF,
            0x76,
            0x00,
            0xFF,
            0x73,
            0x00,
            0xFF,
            0x6F,
            0x00,
            0xFF,
            0x6B,
            0x00,
            0xFF,
            0x67,
            0x00,
            0xFF,
            0x64,
            0x00,
            0xFF,
            0x60,
            0x00,
            0xFF,
            0x5C,
            0x00,
            0xFF,
            0x59,
            0x00,
            0xFF,
            0x55,
            0x00,
            0xFF,
            0x51,
            0x00,
            0xFF,
            0x4D,
            0x00,
            0xFF,
            0x4A,
            0x00,
            0xFF,
            0x46,
            0x00,
            0xFF,
            0x42,
            0x00,
            0xFF,
            0x3F,
            0x00,
            0xFF,
            0x3B,
            0x00,
            0xFF,
            0x37,
            0x00,
            0xFF,
            0x34,
            0x00,
            0xFF,
            0x30,
            0x00,
            0xFF,
            0x2C,
            0x00,
            0xFF,
            0x28,
            0x00,
            0xFF,
            0x25,
            0x00,
            0xFF,
            0x21,
            0x00,
            0xFF,
            0x1D,
            0x00,
            0xFF,
            0x1A,
            0x00,
            0xFF,
            0x16,
            0x00,
            0xFE,
            0x12,
            0x00,
            0xFA,
            0x0F,
            0x00,
            0xF5,
            0x0B,
            0x00,
            0xF1,
            0x07,
            0x00,
            0xEC,
            0x03,
            0x00,
            0xE8,
            0x00,
            0x00,
            0xE3,
            0x00,
            0x00,
            0xDE,
            0x00,
            0x00,
            0xDA,
            0x00,
            0x00,
            0xD5,
            0x00,
            0x00,
            0xD1,
            0x00,
            0x00,
            0xCC,
            0x00,
            0x00,
            0xC8,
            0x00,
            0x00,
            0xC3,
            0x00,
            0x00,
            0xBF,
            0x00,
            0x00,
            0xBA,
            0x00,
            0x00,
            0xB6,
            0x00,
            0x00,
            0xB1,
            0x00,
            0x00,
            0xAC,
            0x00,
            0x00,
            0xA8,
            0x00,
            0x00,
            0xA3,
            0x00,
            0x00,
            0x9F,
            0x00,
            0x00,
            0x9A,
            0x00,
            0x00,
            0x96,
            0x00,
            0x00,
            0x91,
            0x00,
            0x00,
            0x8D,
            0x00,
            0x00,
            0x88,
            0x00,
            0x00,
            0x84,
            0x00,
            0x00,
            0x7F,
            0x00,
            0x00,
        ]
    )

    def __init__(self, *args, no_of_planes=3, width, height, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_of_planes = no_of_planes
        lut_np = cp.frombuffer(self.JET_LUT_U8, dtype=cp.uint8).reshape(256, 3)
        self._jet_lut = cp.asarray(lut_np)  # GPU resident

        self._width = width
        self._height = height
        self._save = 1

    def setup(self, spec):
        logging.info("ADTFUnpackOp setup")
        spec.input("input")
        spec.output("output")

    def start(self):
        pass

    def stop(self):
        pass

    def converttojetimage(self, depth_u16):
        # depth_u16: (H, W), uint16

        depth_norm = cp.clip(
            (depth_u16.astype(cp.float32) / 4000.0) * 255, 0, 255
        ).astype(cp.uint8)

        rgb = self._jet_lut[depth_norm]
        # shape: (H, W, 3), dtype=uint8
        return rgb

    def convert_to_grayscale(self, image):
        # Normalize the depth image to the range 0 to 255
        # 1. First, normalize the values from 0 to 65535 to 0 to 1
        image_normalized = cp.clip(
            image.astype(cp.float32) * 255 / 4096, 0, 255
        ).astype(cp.uint8)
        # 2. Scale it to the range 0 to 255 (for 8-bit grayscale)
        # image_grayscale = cp.clip(image_normalized * 255, 0, 255).astype(cp.uint8)
        image_grayscale = cp.repeat(image_normalized[:, :, None], 3, axis=2)
        return image_grayscale

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("input")
        msg = in_message.get("")
        cp_frame = cp.asarray(msg)
        cp_frame_u8 = (cp_frame >> 8).astype(cp.uint8)

        raw = cp_frame_u8.reshape(
            self._height, self._width, 5
        )  # 2 bytes depth, 1 byte conf, 2 bytes ab

        # Extract  the data from the stream
        depth = raw[:, :, 0].astype(cp.uint16) | (raw[:, :, 1].astype(cp.uint16) << 8)
        conf16 = raw[:, :, 2].astype(cp.uint16) << 8
        active_brightness = raw[:, :, 3].astype(cp.uint16) | (
            raw[:, :, 4].astype(cp.uint16) << 8
        )

        if self._save == 1:
            # dump or save once frame of data, executed only once
            cp_frame_u8.astype("uint8").tofile("dump.bin")
            depth.astype("uint16").tofile("depth.bin")
            conf16.astype("uint16").tofile("conf.bin")
            active_brightness.astype("uint16").tofile("ab.bin")
            self._save = 0

        depth_c = self.converttojetimage(depth)
        active_brightness_c = self.convert_to_grayscale(active_brightness)
        conf_c = self.convert_to_grayscale(conf16)

        if self._no_of_planes == 1:
            op_output.emit(
                {"Depth": cp_frame_u8}, "output"
            )  # CHECK: This is for raw data passing
        elif self._no_of_planes == 2:
            op_output.emit(
                {"Depth": depth_c, "ActiveBrightness": active_brightness_c}, "output"
            )
        elif self._no_of_planes == 3:
            op_output.emit(
                {
                    "Depth": depth_c,
                    "ActiveBrightness": active_brightness_c,
                    "Conf": conf_c,
                },
                "output",
            )


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        ibv_name,
        ibv_port,
        adcam_inst,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._adcam_inst = adcam_inst
        self._frame_limit = frame_limit

    def compose(self):
        logging.info("compose")
        logging.info("Phani - Entering compose")
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

        self._adcam_inst.set_mipi()
        self._adcam_inst.set_mode()
        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._adcam_inst._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._adcam_inst._height,
            num_blocks=2,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._adcam_inst.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        logging.info(f"{frame_size=}")
        frame_context = self._cuda_context

        if self._ibv_name is not None:
            receiver_operator = hololink_module.operators.RoceReceiverOp(
                self,
                condition,
                name="receiver",
                frame_size=frame_size,
                frame_context=frame_context,
                ibv_name=self._ibv_name,
                ibv_port=self._ibv_port,
                hololink_channel=self._hololink_channel,
                device=self._adcam_inst,
            )
        else:
            receiver_operator = hololink_module.operators.LinuxReceiverOperator(
                self,
                condition,
                name="receiver",
                frame_size=frame_size,
                frame_context=frame_context,
                hololink_channel=self._hololink_channel,
                device=self._adcam_inst,
            )

        ADIToF_data = ADTFUnpackOp(
            self,
            name="ADIToF_data",
            no_of_planes=3,
            width=512,
            height=512,
        )

        left_spec = holoscan.operators.HolovizOp.InputSpec(
            "Depth", holoscan.operators.HolovizOp.InputType.COLOR
        )
        left_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
        left_spec_view.offset_x = 0
        left_spec_view.offset_y = 0
        left_spec_view.width = 0.33
        left_spec_view.height = 1
        left_spec.views = [left_spec_view]

        center_spec = holoscan.operators.HolovizOp.InputSpec(
            "ActiveBrightness", holoscan.operators.HolovizOp.InputType.COLOR
        )
        center_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
        center_spec_view.offset_x = 0.33
        center_spec_view.offset_y = 0
        center_spec_view.width = 0.33
        center_spec_view.height = 1
        center_spec.views = [center_spec_view]

        right_spec = holoscan.operators.HolovizOp.InputSpec(
            "Conf", holoscan.operators.HolovizOp.InputType.COLOR
        )
        right_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
        right_spec_view.offset_x = 0.66
        right_spec_view.offset_y = 0
        right_spec_view.width = 0.34
        right_spec_view.height = 1
        right_spec.views = [right_spec_view]

        window_height = 1920
        window_width = 2048  # for the pair
        window_title = "ADI ToF Player"
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            # tensors=[left_spec],
            # tensors=[left_spec, center_spec],
            tensors=[left_spec, center_spec, right_spec],
            height=window_height,
            width=window_width,
            window_title=window_title,
        )

        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(csi_to_bayer_operator, ADIToF_data, {("output", "input")})
        self.add_flow(ADIToF_data, visualizer, {("output", "receivers")})


def int_or_none(value):
    if value == "None":
        return None
    return int(value)


def main():
    # Get a handle to the Hololink port we're connected to.
    parser = argparse.ArgumentParser(
        description="ADITOF Holoscan application parsing arguments"
    )

    # Define arguments
    parser.add_argument(
        "--resetAdcam",
        "-r",
        type=int,
        default=0,
        required=False,
        help="Power on Reset ADCAM module",
    )
    parser.add_argument(
        "--capture",
        "-c",
        type=int,
        default=0,
        required=False,
        help="Capture ADCAM streams",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose mode"
    )
    parser.add_argument(
        "--resetOnly",
        "-RO",
        type=int,
        default=0,
        required=False,
        help="Soft Reset ADCAM module",
    )
    parser.add_argument(
        "--getStatus",
        "-gs",
        type=int,
        default=0,
        required=False,
        help="Get status part of debug",
    )

    # Parse arguments
    # args = parser.parse_args()

    parser.add_argument(
        "--frame-limit",
        type=int_or_none,
        default=300,
        help="Exit after receiving this many frames",
    )

    infiniband_devices = hololink_module.infiniband_devices()

    #  check if we have infiniband_devices or regular linux network
    if infiniband_devices and len(infiniband_devices) > 0:

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
    else:
        logging.info("No Infiniband devices found, using linux player")
        parser.add_argument(
            "--ibv-name",
            default=None,
            help="Network device to use",
        )
        parser.add_argument(
            "--ibv-port",
            type=int,
            default=0,
            help="Port number of device",
        )

    args = parser.parse_args()

    hololink_module.logging_level(2)
    logging.info("Initializing.")

    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    logging.info("starting")
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip="192.168.0.2")
    logging.info(f"{channel_metadata=}")
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Instantiate the adcam_inst itself; CAM_I2C_BUS is the appropriate bus enable setting
    # for the I2C controller our adcam_inst is attached to
    adcam_inst = adcam.adcam(
        hololink_channel, hololink_module.CAM_I2C_BUS, channel_metadata
    )

    # Establish a connection to the hololink device
    hololink = hololink_channel.hololink()
    hololink.start()

    if args.resetAdcam == 1:
        logging.info("Doing the full Reset including power on sequence")
        adcam_inst.adcam_reset_power_on(hololink, hololink_channel, channel_metadata)

    if args.resetOnly == 1:
        logging.info("Performing ONLY Reset - NOT doing FULL Power on reset")
        adcam_inst.adcam_Only_reset(hololink, hololink_channel, channel_metadata)

    # add FW upgrade as well

    # check if the chip exists
    if adcam_inst.probe_adcam_adtf3175() != 1:
        logging.error("No ADCAM ADTF3175 found, connect ADCAM, reset and try again")
        hololink.stop()
        exit()

    # Fetch the device version.
    if args.getStatus == 1:
        logging.debug("Getting only status")
        adcam_inst.get_status()

    version = adcam_inst.get_fw_version()
    logging.info(f"{version=}")

    if args.capture == 1:
        # Set up the application
        application = HoloscanApplication(
            False,
            True,
            cu_context,
            cu_device_ordinal,
            hololink_channel,
            args.ibv_name,
            args.ibv_port,
            adcam_inst,
            args.frame_limit,
        )
    if args.capture == 1:
        # adcam_inst.set_mode ()
        application.run()
    elif args.capture == 2:
        logging.debug("Force stop capture..")
        adcam_inst.stream_off()

    hololink.stop()


if __name__ == "__main__":
    main()
