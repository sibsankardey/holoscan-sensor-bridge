# ==============================================================================
# Holoscan Sensor Bridge - ADI IMU Operator
# ==============================================================================

if(NOT HOLOLINK_BUILD_ONLY_NATIVE)
# 1. Build the native C++ library SHARED
add_library(hololink_adi_imu_op STATIC
    adi_imu_op.cpp
)

# Link against Holoscan and Hololink core libraries
target_link_libraries(hololink_adi_imu_op
    PUBLIC
        hololink
        holoscan::core
        holoscan::pybind11
    	hololink
)

# ---------------------------------------------------------------------------
# 2. Build the Python Bindings using the Repo's Internal Pybind11
# ---------------------------------------------------------------------------
# This is the magic line that fixes the Docker build:
include(hololink_deps/pybind11)

# Create the Python module
pybind11_add_module(_adi_imu_op MODULE
    python/adi_imu_op_python.cpp
)
# This forces Pybind11 to drop Python version tag so it works across ROS upgrades
set_target_properties(_adi_imu_op PROPERTIES SUFFIX ".so")

# Link the Pybind wrapper to our native C++ library and Holoscan
target_link_libraries(_adi_imu_op
    PRIVATE
        hololink_adi_imu_op
        holoscan::core
)

# ---------------------------------------------------------------------------
# 3. Copy Initialization & Installation Rules
# ---------------------------------------------------------------------------
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
    COPYONLY
)

#install(TARGETS hololink_adi_imu_op DESTINATION lib)
#install(TARGETS _adi_imu_op DESTINATION hololink/operators/adi_imu)
#install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py DESTINATION hololink/operators/adi_imu)

install(TARGETS hololink_adi_imu_op
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS _adi_imu_op
        LIBRARY DESTINATION ${HOLOLINK_PYTHON_INSTALL_DIR}/hololink/operators/adi_imu)
install(FILES python/__init__.py
        DESTINATION ${HOLOLINK_PYTHON_INSTALL_DIR}/hololink/operators/adi_imu)
# ---------------------------------------------------------------------------
# 4. Unified Build Target
# ---------------------------------------------------------------------------
# This creates a single 'make' command that safely builds both the
# C++ hardware driver and the Python Pybind11 wrapper together.
add_custom_target(adi_imu
    DEPENDS hololink_adi_imu_op _adi_imu_op
)

endif() # NOT HOLOLINK_BUILD_ONLY_NATIVE
