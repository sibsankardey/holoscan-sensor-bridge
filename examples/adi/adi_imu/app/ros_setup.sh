#!/bin/bash
#!/bin/bash
set -e

echo "========================================"
echo "ROS 2 Installer"
echo "========================================"
echo
echo "This script will:"
echo "  - Detect Ubuntu version"
echo "  - Install ROS 2 Humble (22.04)"
echo "    or ROS 2 Jazzy (24.04)"
echo "  - Install CycloneDDS"
echo "  - Configure ROS environment"
echo
read -rp "Do you want to continue? [Y/N]: " INSTALL_ROS

case "$INSTALL_ROS" in
    Y|y)
        echo "Proceeding with ROS installation..."
        ;;
    *)
        echo "Installation cancelled."
        exit 0
        ;;
esac

# --------------------------------------------------------------------------
# Detect Ubuntu Version
# --------------------------------------------------------------------------

echo "Detecting Ubuntu release..."
source /etc/os-release

case "${VERSION_ID}" in
    "22.04")
        ROS_DISTRO="humble"
        ;;
    "24.04")
        ROS_DISTRO="jazzy"
        ;;
    *)
        echo "Unsupported Ubuntu version: ${VERSION_ID}"
        exit 1
        ;;
esac

echo "Installing ROS2 ${ROS_DISTRO}..."

export DEBIAN_FRONTEND=noninteractive


echo "Installing ROS2 ${ROS_DISTRO} on Ubuntu ${UBUNTU_VERSION}"

# --------------------------------------------------------------------------
# Locale Setup
# --------------------------------------------------------------------------

apt-get update

apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common

locale-gen en_US en_US.UTF-8

update-locale \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8

export LANG=en_US.UTF-8

# --------------------------------------------------------------------------
# ROS Repository
# --------------------------------------------------------------------------

add-apt-repository universe -y

curl -sSL \
    https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu \
${UBUNTU_CODENAME} main" \
> /etc/apt/sources.list.d/ros2.list

apt-get update

# --------------------------------------------------------------------------
# Install ROS Core
# --------------------------------------------------------------------------

apt-get install -y \
    ros-${ROS_DISTRO}-ros-core \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp

# Optional robotics packages
apt-get install -y \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-rclcpp \
    ros-${ROS_DISTRO}-rclpy

# IMU tools if desired
apt-get install -y \
    ros-${ROS_DISTRO}-imu-tools || true

# RViz only if GUI is needed
# apt-get install -y ros-${ROS_DISTRO}-rviz2

apt-get clean
rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# CycloneDDS Configuration
# --------------------------------------------------------------------------

mkdir -p /etc/cyclonedds

cat >/etc/cyclonedds/config.xml <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <AllowMulticast>true</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
  </Domain>
</CycloneDDS>
EOF

# --------------------------------------------------------------------------
# Environment Variables
# --------------------------------------------------------------------------

grep -q "ROS_DOMAIN_ID" ~/.bashrc || \
echo "export ROS_DOMAIN_ID=5" >> ~/.bashrc

grep -q "RMW_IMPLEMENTATION" ~/.bashrc || \
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc

grep -q "CYCLONEDDS_URI" ~/.bashrc || \
echo "export CYCLONEDDS_URI=file:///etc/cyclonedds/config.xml" >> ~/.bashrc

grep -q "/opt/ros/${ROS_DISTRO}/setup.bash" ~/.bashrc || \
echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# --------------------------------------------------------------------------
# Docker Entrypoint
# --------------------------------------------------------------------------

cat >/ros_entrypoint.sh <<EOF
#!/bin/bash
set -e

source /opt/ros/${ROS_DISTRO}/setup.bash

export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-5}
export RMW_IMPLEMENTATION=\${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
export CYCLONEDDS_URI=\${CYCLONEDDS_URI:-file:///etc/cyclonedds/config.xml}

exec "\$@"
EOF

chmod +x /ros_entrypoint.sh

echo ""
echo "=========================================="
echo "ROS2 ${ROS_DISTRO} installation complete"
echo "Ubuntu : ${UBUNTU_VERSION}"
echo "ROS    : ${ROS_DISTRO}"
echo "Domain : 5"
echo "DDS    : CycloneDDS"
echo "=========================================="

source /opt/ros/${ROS_DISTRO}/setup.bash

ros2 doctor --report || true


echo "Note: If you are running this within container, ROS2 will be removed on next container launch..need to rerun this script"
echo "Manually run 'source /opt/ros/${ROS_DISTRO}/setup.bash' on the shell to setup the environment before running ADI IMU capture"
