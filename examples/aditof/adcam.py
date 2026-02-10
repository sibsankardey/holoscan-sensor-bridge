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
import logging
import struct
import time

import hololink as hololink_module


class ADCAMEXPANDER:
    EXPANDER_0_I2C_BUS_ADDRESS = 0x68

    def __init__(self, hololink_channel, hololink_i2c_controller_address, expan_addr):
        # Get handles to these controllers but don't actually talk to them yet
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(hololink_i2c_controller_address)
        self._expan_addr = expan_addr

    def set_register(self, register, value, timeout=None):
        logging.debug(
            "set_register(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint8(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            self._expan_addr,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )
        print("Expander", self._expan_addr, "byte_count = ", read_byte_count)


class polarfireGpio:

    # configurations to test
    configs = [
        "ALL_OUT_L",  # All pins output low
        "ALL_OUT_H",  # All pins output high
        "ALL_IN",  # All pins inputs
        "ODD_OUT_H",  # Odd pins output high, even pins input - use jumper to short & test
        "EVEN_OUT_H",  # Even pins output high, Odd pins input - use jumper to short & test
    ]

    # dictionary for print beautification
    dir = {0: "output", 1: "input"}

    # def __init__(self, fragment, hololink_channel, gpio, *args, **kwargs):
    def __init__(self, hololink_channel, gpio):
        self._hololink = hololink_channel.hololink()
        self._gpio = gpio
        self.pin = 0
        self.test_config = 0

        # how many pins are supported on the platform running the example
        self._supported_pins_number = self._gpio.get_supported_pin_num()

        print("Totel supported GPIOs = ", self._supported_pins_number)

    def setup(self):
        # spec.output("gpio_changed_out")
        # spec.output("test_config_out")

        # set all gpios as output, high - test fast setting via loop
        for i in range(self._supported_pins_number):
            print("Left as it pin =", i)

    def pull_reset_low(self, pin):
        self._gpio.set_direction(self.pin, self._gpio.OUT)
        self._gpio.set_value(self.pin, self._gpio.LOW)

    def pull_reset_high(self, pin):
        self._gpio.set_direction(self.pin, self._gpio.OUT)
        self._gpio.set_value(self.pin, self._gpio.HIGH)


# class adtf3175:
class adcam:
    ADCAM_I2C_BUS_ADDRESS = 0x38
    EXPANDER_0_I2C_BUS_ADDRESS = 0x68
    EXPANDER_1_I2C_BUS_ADDRESS = 0x58

    def __init__(
        self, hololink_channel, hololink_i2c_controller_address, channel_metadata
    ):
        # Get handles to these controllers but don't actually talk to them yet
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(hololink_i2c_controller_address)
        self._width = 2560
        self._height = 512
        self._mode = 3  # mode
        self._pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_8
        self._expander0 = ADCAMEXPANDER(
            hololink_channel,
            hololink_module.CAM_I2C_BUS,
            self.EXPANDER_0_I2C_BUS_ADDRESS,
        )
        self._expander1 = ADCAMEXPANDER(
            hololink_channel,
            hololink_module.CAM_I2C_BUS,
            self.EXPANDER_1_I2C_BUS_ADDRESS,
        )
        self._pf_gpio = polarfireGpio(
            hololink_channel, self._hololink.get_gpio(channel_metadata)
        )

    def get_version(self):
        logging.info("Fatching Chip version")
        REGISTER = 0x0112
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Response = {resp}")

        # Fetch status
        REGISTER = 0x0020
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Response = {resp}")

        return resp

    def bytes_to_uint16_array(self, data: bytes):
        """Convert bytes/bytearray to list of 16-bit unsigned integers (big-endian)."""
        if len(data) % 2 != 0:
            raise ValueError("Data length must be even for 16-bit conversion.")
        return list(struct.unpack(f">{len(data)//2}H", data))

    def registers_to_byte_array(self, register):
        """Convert bytes/bytearray to list of 16-bit unsigned integers (big-endian)."""
        length = (register.bit_length() + 7) // 8

        if length == 0:
            length = 2
        if length % 2 != 0:
            length = length + 1

        byte_array = register.to_bytes(length, byteorder="big")
        return byte_array

    def format_registers(self, reg):
        """Convert odd size register to uint16 list (big-endian)."""
        return self.bytes_to_uint16_array(self.registers_to_byte_array(reg))

    def set_mipi(self):
        logging.info("Setting MIPI speed")
        # REGISTER = 0x00310003 #1.5gbps
        REGISTER = 0x00310004  # 1gbps
        # REGISTER = 0x00310001 #2.5gbps
        self.set_register16_no_response(REGISTER)

        logging.info("Enabling deskew")
        REGISTER = 0x00AB0001
        self.set_register16_no_response(REGISTER)

    def set_mode(self):
        print("Setting QMP mode")
        logging.info("Setting QMP mode")
        REGISTER = 0xDA06280F
        self.set_register16_no_response(REGISTER)

    def read_nvm_config(self):
        logging.info("Reading NVM Config")
        time.sleep(1)
        REGISTER = 0x00190000
        self.set_register16_no_response(REGISTER)

        # Read fw version
        REGISTER = 0xAD002C05000000003100000001000000
        resp = self.set_register16_response(REGISTER, 44)
        logging.info(f"Firmware ID = {resp}")

        # Read NVM header
        REGISTER = 0xAD000013000000001300000001000000
        resp = self.set_register16_response(REGISTER, 44)
        logging.info(f"NVM Header = {resp}")
        print("NVM Header = ", resp)

        # Read fw version
        REGISTER = 0xAD002C05000000003100000001000000
        resp = self.set_register16_response(REGISTER, 44)
        logging.info(f"Firmware ID = {resp}")
        print("FW ID = ", resp)

        # turn off burst mode
        REGISTER = 0xAD000010000000001000000000000000
        REGISTER = 0xAD000010000000001000000001000000
        self.set_register16_no_response(REGISTER)
        logging.info("Reading NVM Config done")

    def get_status(self):
        logging.info("Fatching status")
        # Get Status
        REGISTER = 0x0020
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Chip Status = {resp}")
        print(f"Chip Status = {resp}")
        REGISTER = 0x0038
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"0x0038 Status = {resp}")

    def get_only_status(self):
        logging.info("Fatching status")
        # Get chip ID
        REGISTER = 0x0020
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Chip Status = {resp}")

    def force_stop_burst_mode(self):
        logging.info("Forcing burst mode off")

        # turn off burst mode
        REGISTER = 0xAD000010000000001000000001000000
        self.set_register16_no_response(REGISTER)

        return self.get_status()

    def get_fw_version(self):
        logging.info("Fatching version")

        # Get chip ID
        REGISTER = 0x0112
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Chip ID = {resp}")

        # Get Status
        REGISTER = 0x0020
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Chip Status = {resp}")

        # set to burst mode
        REGISTER = 0x00190000
        self.set_register16_no_response(REGISTER)

        # Read fw version
        REGISTER = 0xAD002C05000000003100000001000000
        resp = self.set_register16_response(REGISTER, 44)
        logging.info(f"Firmware ID = {resp}")

        # turn off burst mode
        REGISTER = 0xAD000010000000001000000001000000
        self.set_register16_no_response(REGISTER)

        return self.get_status()

    def get_chip_status(self):
        logging.info("Fetching status")
        # Get Status
        REGISTER = 0x0020
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Chip Status = {resp}")

        REGISTER = 0x0038
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"0x0038 Status = {resp}")

    def set_register(self, register, value, timeout=None):
        logging.debug(
            "set_register(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint8(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            self.ADCAM_I2C_BUS_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def set_register16_no_response(self, register, resp_len=0, timeout=None):
        try:
            write_bytes = bytearray(100)
            uint16_array = self.format_registers(register)
            logging.info("ADCAM REGISTER NORESPREQD =(0x%X))" % (register))

            serializer = hololink_module.Serializer(write_bytes)
            for i in range(0, len(uint16_array), 1):
                chunk = uint16_array[i]
                # The inner function handles the format string correctly for each chunk
                # serializer.append_uint16_be(struct.unpack('>h', chunk))
                serializer.append_uint16_be(chunk)

            read_byte_count = 0
            self._i2c.i2c_transaction(
                self.ADCAM_I2C_BUS_ADDRESS,
                serializer.data(),
                read_byte_count,
                timeout=timeout,
            )
            print("ADCAM:", hex(register), "Response", None)
            # logging.info("set_register16_no_response write =",serializer.data())
        except AttributeError as e:
            logging.info(f"[ERROR] Attribute missing or invalid object used: {e}")
            return None
        except ValueError as e:
            logging.info(f"[ERROR] Value or data format issue: {e}")
            return None
        except OSError as e:
            logging.info(f"[ERROR] I2C communication failed (OS error): {e}")
            return None
        except Exception as e:
            logging.info(f"[ERROR] Unexpected failure during I2C transaction: {e}")
            return None

    def set_register16_response(self, register, resp_len, timeout=None):
        try:
            write_bytes = bytearray(100)
            uint16_array = self.format_registers(register)
            logging.info("ADCAM REGISTER RESPREQD =(0x%X))" % (register))
            serializer = hololink_module.Serializer(write_bytes)
            for i in range(0, len(uint16_array), 1):
                chunk = uint16_array[i]
                # The inner function handles the format string correctly for each chunk
                serializer.append_uint16_be(chunk)
            read_byte_count = resp_len
            adi_tof_timeout = hololink_module.Timeout(30, retry_s=0.2)
            reply = self._i2c.i2c_transaction(
                self.ADCAM_I2C_BUS_ADDRESS,
                serializer.data(),
                read_byte_count,
                timeout=adi_tof_timeout,
            )
            deserializer = hololink_module.Deserializer(reply)
            deserializer.next_uint8()
            logging.info(f"ADCAM REGISTER REPLY from Chip = {reply}")
            print("ADCAM:", hex(register), "Response", reply)
            return reply
        except AttributeError as e:
            logging.info(f"[ERROR] Attribute missing or invalid object used: {e}")
            return reply
        except ValueError as e:
            logging.info(f"[ERROR] Value or data format issue: {e}")
            return reply
        except OSError as e:
            logging.info(f"[ERROR] I2C communication failed (OS error): {e}")
            return None
        except Exception as e:
            logging.info(f"[ERROR] Unexpected failure during I2C transaction: {e}")
            return None

    def stream_on(self):
        print("Setting Clock continuous mode")
        logging.info("Setting Clock continuous mode in stream_on")
        CONT_MODE = 0x00A90001
        self.set_register16_no_response(CONT_MODE)
        time.sleep(0.2)
        print("Turning ON Streaming")
        logging.info("Turning ON Streaming")
        STREAM_MODE = 0x00AD00C5
        self.set_register16_no_response(STREAM_MODE)
        time.sleep(0.2)
        self.get_status()

    def stream_off(self):
        print("Turning OFF Streaming")
        logging.info("Turning OFF Streaming")
        STREAM_MODE = 0x000C0002
        self.set_register16_no_response(STREAM_MODE)
        time.sleep(0.2)
        self.get_status()

    def get_ChipID(self):
        # Get chip ID
        REGISTER = 0x0112
        resp = self.set_register16_response(REGISTER, 2)
        print((f"Chip ID = {resp}"))
        logging.info(f"Chip ID = {resp}")

    def get_Status(self):
        # Get Status
        REGISTER = 0x0020
        resp = self.set_register16_response(REGISTER, 2)
        print((f"resp = {resp}"))
        logging.info(f"Chip Status = {resp}")

    def get_ClockContinuousMode(self):
        # Get Status
        REGISTER = 0x00AA
        resp = self.set_register16_response(REGISTER, 2)
        print((f"Clock continuous mode =  {resp}"))
        logging.info(f"Clock continuous mode = {resp}")
        return resp

    def adcam_reset_power_on(self, hololink, hololink_channel, channel_metadata):

        logging.info("Resetting ADCAM")
        self._pf_gpio.pull_reset_low(0)
        self._expander0.set_register(0x0, 0x0)  # Force all the Expandoer 0 bits as 0
        self._expander1.set_register(0x0, 0x0)  # Force all the Expandoer 1 bits as 0

        # O1 (EN_0P8) => 1 //158  Power enable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        # Expander0.get_register(0x02) #This should turn on the DS1 LED ON
        self._expander0.set_register(0x02, 0x02)  # This should turn on the DS1 LED ON
        logging.info("Check DS1 LED - it should be ON. This will be on ")
        # Expander0.get_register(0x02) #This should turn on the DS1 LED ON
        time.sleep(0.2)  # LED ON for 10 sec
        self._expander0.set_register(0x0, 0x0)  # This should turn OFF the DS1 LED
        logging.info("DS1 LED turned off")
        # Checking DONE!

        # P5  (HOST_IO_SEL) => 1 //114 //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x20, 0x20)  # E0 = 0x20
        # O8 (HOST_IO_DIR) => 1 //117 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        # O11 (FSYNC_DIR) => 1 //120 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        self._expander1.set_register(0x9, 0x9)  # E1 = 0x09

        # RST to low
        # Add code to make RST GPIO Low

        # O0 (EN_1P8) => 0 //127. Power disable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        # O1 (EN_0P8) => 0 //130. Power disable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x20, 0x20)  # E0 = 0x20 //No change. Remains same
        time.sleep(0.2)  # Pauses execution for 0.2 seconds (200 milliseconds)

        # P3  (I2CM_SET) => 0 //135. Enable SPI for imager to pulsatrix //O7 O6 P5 P4 P3 P2 O1 O0 bits
        # O6 (ISP_BS0) => 0 //138  - Boot strap pins //O7 O6 P5 P4 P3 P2 O1 O0 bits
        # O7 (ISP_BS1) => 0 //141 - Boot strap pins //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x20, 0x20)  # E0 = 0x20 //No change. Remains same

        # O9 (ISP_BS4) => 0 //143 - Boot strap pins //O15 O14 O13 O12 O11 O10 O9 O8 bits
        # O10 (ISP_BS5) => 0 //147 - Boot strap pins //O15 O14 O13 O12 O11 O10 O9 O8 bits
        self._expander1.set_register(0x9, 0x9)  # E1 = 0x09 //No change. Remains same

        # O0 (EN_1P8) => 1 //153. Power enable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x21, 0x21)  # E0 = 0x21
        time.sleep(0.2)  # Pauses execution for 0.2 seconds (200 milliseconds)

        # O1 (EN_0P8) => 1 //158  Power enable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x23, 0x23)  # E0 = 0x23
        time.sleep(0.2)  # Pauses execution for 0.2 seconds (200 milliseconds)

        # O14 (EN_VSYS) => 1 //163 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        # O13 (EN_VAUX_LS) => 1 //166 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        # O12 (EN_VAUX) => 1 //169 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        self._expander1.set_register(0x79, 0x79)  # E1 = 0x79
        time.sleep(0.2)  # Pauses execution for 0.2 seconds (200 milliseconds)
        self._pf_gpio.pull_reset_high(0)

        logging.info("booting up ADSD, wait for 10 seconds")
        time.sleep(10)  # Bootup ADSD3500

    def adcam_Only_reset(self, hololink, hololink_channel, channel_metadata):
        # pf_gpio = polargpio.polarfireGpio(hololink_channel, hololink.get_gpio(channel_metadata) )
        print("ADCAM - Making Reset LOW ONLY")
        self._pf_gpio.pull_reset_low(0)

        time.sleep(1)  # Pauses execution for 0.2 seconds (200 milliseconds)
        print("ADCAM - Making Reset HIGH ONLY")
        self._pf_gpio.pull_reset_high(0)

        print("Waiting 10 secs after reset")
        time.sleep(10)  # Bootup ADSD3500

    def configure_converter(self, converter):
        # where do we find the first received byte?
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            self._pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
        converter.configure(
            start_byte,
            received_line_bytes,
            self._width,
            self._height,
            self._pixel_format,
        )

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        return hololink_module.sensors.csi.BayerFormat.RGGB

    def start(self):
        """Setting and checking Clock continuous mode"""
        self.get_status()
        # time.sleep(0.6)
        # mode = self.get_ClockContinuousMode()
        if 0:  # mode[0] == 1:
            print("Contineous clock mode already enabled")
            logging.info("Contineous clock mode already enabled")
        else:
            print("Setting Clock continuous mode in start")
            logging.info("Setting Clock continuous mode")
            CONT_MODE = 0x00A90001
            self.set_register16_no_response(CONT_MODE)
            time.sleep(0.6)
            # Reading the Clock mode for confirmation...
            self.get_ClockContinuousMode()
            time.sleep(0.6)

        """Start Streaming"""
        # self.set_register(0x100, 0x01),
        logging.info("Turning ON Streaming")
        print("Turning ON Streaming TS in sec=", int(time.time()))
        STREAM_MODE = 0x00AD00C5
        self.set_register16_no_response(STREAM_MODE)
        time.sleep(0.2)
        self.get_status()
        # self.set_register16_response(0x0058, 2) # read FSYNC status, enabled for debug

    def stop(self):
        """Stop Streaming"""
        logging.info("Turning OFF Streaming")
        print("Turning OFF Streaming TS in sec=", int(time.time()))
        self.set_register16_response(0x0058, 2)  # read FSYNC status
        STREAM_MODE = 0x000C0002
        self.set_register16_no_response(STREAM_MODE)
        time.sleep(0.2)
        self.get_status()
