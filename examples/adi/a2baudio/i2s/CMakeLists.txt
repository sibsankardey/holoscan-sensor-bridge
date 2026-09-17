if(NOT HOLOLINK_BUILD_ONLY_NATIVE)
add_library(i2s_receiver STATIC
  i2s_receiver_op.cpp
  )

set_property(TARGET i2s_receiver PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::i2s_receiver ALIAS i2s_receiver)

target_include_directories(i2s_receiver
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(i2s_receiver
  PRIVATE
    hololink::core
    holoscan::core
    hololink::operators::linux_receiver_op # Required for base class inheritance
  )

# Installation of the i2s_receiver operator
set_target_properties(i2s_receiver PROPERTIES EXPORT_NAME operators::i2s_receiver)
install(TARGETS i2s_receiver
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/i2s_receiver_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/i2s_receiver
        COMPONENT hololink-operators)
endif() # NOT HOLOLINK_BUILD_ONLY_NATIVE
