add_library(audio_beamformer STATIC
  audio_beamformer_op.cpp
  audio_beamformer_op.cu
  )

set_target_properties(audio_beamformer PROPERTIES 
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  )

add_library(hololink::operators::audio_beamformer ALIAS audio_beamformer)

target_include_directories(audio_beamformer
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(audio_beamformer
  PRIVATE
    hololink::core
    holoscan::core
    CUDA::cudart
  )

install(TARGETS audio_beamformer
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)
