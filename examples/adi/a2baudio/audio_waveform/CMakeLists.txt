add_library(audio_waveform STATIC
  audio_waveform_op.cu
  )

# For CUDA mixing with GXF/Holoscan, separable compilation is strongly recommended
set_target_properties(audio_waveform PROPERTIES 
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  )

add_library(hololink::operators::audio_waveform ALIAS audio_waveform)

target_include_directories(audio_waveform
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(audio_waveform
  PRIVATE
    hololink::core
    holoscan::core
    CUDA::cudart # Needed for the CUDA Runtime API used in the kernel
    cufft
  )

# Installation of the audio_waveform operator
set_target_properties(audio_waveform PROPERTIES EXPORT_NAME operators::audio_waveform)
install(TARGETS audio_waveform
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/audio_waveform_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/audio_waveform
        COMPONENT hololink-operators)
