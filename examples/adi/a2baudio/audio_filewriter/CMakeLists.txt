
if(NOT HOLOLINK_BUILD_ONLY_NATIVE)
add_library(audio_filewriter STATIC
  audio_file_writer_op.cpp
  )

set_property(TARGET audio_filewriter PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::audio_filewriter ALIAS audio_filewriter)

target_include_directories(audio_filewriter
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(audio_filewriter
  PRIVATE
    hololink::core
    holoscan::core
    hololink::operators::linux_receiver_op # Required for base class inheritance
  )

# Installation of the audio_filewriter operator
set_target_properties(audio_filewriter PROPERTIES EXPORT_NAME operators::audio_filewriter)
install(TARGETS audio_filewriter
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/audio_file_writer_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/audio_filewriter
        COMPONENT hololink-operators)

endif() # NOT HOLOLINK_BUILD_ONLY_NATIVE
