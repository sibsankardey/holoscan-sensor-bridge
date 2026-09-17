import os
import tempfile

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # 1. RViz configuration (Bypassing the TF Tree)
    rviz_config_content = """
Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Grid
      Enabled: true
      Name: Grid
    - Class: rviz_imu_plugin/Imu
      Enabled: true
      Name: Imu
      Topic:
        Value: /imu/data
        Reliability Policy: Reliable
        Durability Policy: Volatile
      Box:
        Value: true
        Scale:
          X: 1.0
          Y: 1.0
          Z: 1.0
  Global Options:
    Fixed Frame: imu_link
  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 5.0
      Name: Current View
      Pitch: 0.78
      Yaw: 0.78
      Focal Point:
        X: 0.0
        Y: 0.0
        Z: 0.0
"""

    rviz_config_file = os.path.join(tempfile.gettempdir(), "working_imu_config.rviz")
    with open(rviz_config_file, "w") as f:
        f.write(rviz_config_content)

    return LaunchDescription(
        [
            # 2. Madgwick Filter Node (Handling the Best Effort hardware)
            Node(
                package="imu_filter_madgwick",
                executable="imu_filter_madgwick_node",
                name="imu_filter",
                parameters=[
                    {
                        "use_mag": False,
                        "qos_overrides./imu/data_raw.subscription.reliability": "best_effort",
                        "qos_overrides./imu/data_raw.subscription.durability": "volatile",
                    }
                ],
            ),
            # 3. RViz2 (Loading the perfect config)
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config_file],
                output="screen",
            ),
        ]
    )
