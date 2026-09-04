# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Portions Copyright (c) 2026 Analog Devices, Inc.
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

set(ADIA2B_INSTALL_TARGETS)
if(NOT HOLOLINK_BUILD_ONLY_NATIVE)

add_executable(audio_viz_disp
  audio_viz.cpp
)

target_link_libraries(audio_viz_disp
  PRIVATE
    hololink
    holoscan::core
    holoscan::ops::holoviz
    hololink::operators::linux_receiver_op
    CUDA::cuda_driver
    fmt::fmt
    # operators used
    hololink::operators::audio_waveform
    hololink::operators::audio_beamformer
    hololink::operators::audio_filewriter
)
	

list(APPEND ADIA2B_INSTALL_TARGETS audio_viz_disp)


endif() # NOT HOLOLINK_BUILD_ONLY_NATIVE

# Installation of the examples
install(TARGETS ${ADIA2B_INSTALL_TARGETS}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT hololink-examples)
