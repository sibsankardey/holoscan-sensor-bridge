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

set(ADITOF_INSTALL_TARGETS)

if(HOLOLINK_BUILD_ROCE)
    set(RECEIVER_LIB hololink::operators::roce_receiver)
else()
    set(RECEIVER_LIB hololink::operators::linux_receiver_op)
endif()

if(NOT HOLOLINK_BUILD_ONLY_NATIVE)

find_package(yaml-cpp REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

add_executable(adcam_player
  adcam_player.cpp
  adcam_lib.cpp
  adcam_calibration.cpp
  adcam_unpack_op.cpp
  adcam_unpack_op.cu
  programmer.cpp
  adsd3500_flash.cpp
  compute_crc.cpp
  crc_table.cpp
  )

if(HOLOLINK_ENABLE_ROCE)
    target_compile_definitions(adcam_player PRIVATE USE_ROCE_RECEIVER=1)
else()
    target_compile_definitions(adcam_player PRIVATE USE_ROCE_RECEIVER=0)
endif()

include(hololink_deps/pybind11)
target_link_libraries(adcam_player
  PRIVATE
    hololink
    hololink::operators::csi_to_bayer
    hololink::operators::image_processor
    ${RECEIVER_LIB}
    holoscan::core
    holoscan::ops::bayer_demosaic
    holoscan::ops::holoviz
    holoscan::viz::imgui
    holoscan::pybind11
    pybind11::embed
    yaml-cpp
    CURL::libcurl
    OpenSSL::SSL
    OpenSSL::Crypto
  )

list(APPEND ADITOF_INSTALL_TARGETS adcam_player)


endif() # NOT HOLOLINK_BUILD_ONLY_NATIVE

# Installation of the examples
install(TARGETS ${ADITOF_INSTALL_TARGETS}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT hololink-examples)
