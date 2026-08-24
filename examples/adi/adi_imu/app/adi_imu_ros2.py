import argparse
import math
import os
import sys

import rclpy

# from _adi_imu_op import create_imu_operator
from hololink.operators.adi_imu import create_imu_operator
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, Temperature


class NativeRos2PublisherOp(Operator):
    def __init__(
        self,
        fragment,
        imu_model="ADIS16505-2",
        model_config=None,
        *args,
        local_gravity=9.80665,
        calibrate_time=0.0,
        **kwargs,
    ):

        self.imu_model = imu_model

        if model_config is None:
            raise ValueError(f"Scaling parameters for {imu_model} not found in YAML.")

        self.gravity = local_gravity
        self.gyro_scale = model_config["gyro_scale"]
        self.accel_scale = model_config["accel_scale"]
        self.temp_scale = model_config["temp_scale"]
        self.temp_offset = model_config["temp_offset"]

        self.calibrate_time = calibrate_time
        self.is_calibrating = self.calibrate_time > 0.0
        self.calib_start_time = None
        self.calib_samples = 0

        self.sum_gx = 0.0
        self.sum_gy = 0.0
        self.sum_gz = 0.0
        self.sum_ax = 0.0
        self.sum_ay = 0.0
        self.sum_az = 0.0

        rclpy.init()
        self.node = Node("adi_imu_holoscan_node")

        madgwick_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.imu_pub = self.node.create_publisher(Imu, "imu/data_raw", madgwick_qos)
        self.temp_pub = self.node.create_publisher(
            Temperature, "imu/temperature", madgwick_qos
        )

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("imu_raw")
        spec.input("temp_raw")

    def finish_calibration(self):
        mean_gx = self.sum_gx / self.calib_samples
        mean_gy = self.sum_gy / self.calib_samples
        mean_gz = self.sum_gz / self.calib_samples
        mean_ax = self.sum_ax / self.calib_samples
        mean_ay = self.sum_ay / self.calib_samples
        mean_az = self.sum_az / self.calib_samples

        bias_ax = mean_ax
        bias_ay = mean_ay
        bias_az = mean_az

        if abs(mean_ax) > 5.0:
            bias_ax = mean_ax - math.copysign(self.gravity, mean_ax)
            grav_axis = "X"
        elif abs(mean_ay) > 5.0:
            bias_ay = mean_ay - math.copysign(self.gravity, mean_ay)
            grav_axis = "Y"
        else:
            bias_az = mean_az - math.copysign(self.gravity, mean_az)
            grav_axis = "Z"

        print("\n" + "=" * 60)
        print(
            f" CALIBRATION COMPLETE: {self.calib_samples} samples over {self.calibrate_time} seconds"
        )
        print(f" Detected Gravity on Axis: {grav_axis}")
        print("=" * 60)

        print("\n[SOFTWARE CALIBRATION FLAGS]")
        print("Run the node with these arguments to apply IMU bias registers:")
        print(
            f"--calib_gx {mean_gx:.8f} --calib_gy {mean_gy:.8f} --calib_gz {mean_gz:.8f} \\"
        )
        print(
            f"--calib_ax {bias_ax:.8f} --calib_ay {bias_ay:.8f} --calib_az {bias_az:.8f}"
        )
        print("\n" + "=" * 60 + "\n")

        os._exit(0)

    def compute(self, op_input, op_output, context):
        imu_str = op_input.receive("imu_raw")
        temp_str = op_input.receive("temp_raw")

        if not imu_str or not temp_str:
            return

        imu_parts = imu_str.split(",")
        if len(imu_parts) == 7:
            hw_timestamp = float(imu_parts[0])

            gx = int(imu_parts[1])
            gy = int(imu_parts[2])
            gz = int(imu_parts[3])
            ax = int(imu_parts[4])
            ay = int(imu_parts[5])
            az = int(imu_parts[6])

            phys_gx = float(gx * self.gyro_scale)
            phys_gy = float(gy * self.gyro_scale)
            phys_gz = float(gz * self.gyro_scale)
            phys_ax = float(ax * self.accel_scale)
            phys_ay = float(ay * self.accel_scale)
            phys_az = float(az * self.accel_scale)

            if self.is_calibrating:
                if self.calib_start_time is None:
                    self.calib_start_time = hw_timestamp
                    self.last_printed_sec = -1
                    print(
                        f"\n[CALIBRATION STARTED] Averaging data for {self.calibrate_time} seconds..."
                    )
                    print("!!! DO NOT TOUCH OR VIBRATE THE IMU !!!\n")

                self.sum_gx += phys_gx
                self.sum_gy += phys_gy
                self.sum_gz += phys_gz
                self.sum_ax += phys_ax
                self.sum_ay += phys_ay
                self.sum_az += phys_az
                self.calib_samples += 1

                elapsed = hw_timestamp - self.calib_start_time
                remaining = int(math.ceil(self.calibrate_time - elapsed))

                if remaining != self.last_printed_sec and remaining >= 0:
                    sys.stdout.write(
                        f"\r[CALIBRATING] Keep IMU still... {remaining} seconds remaining. "
                    )
                    sys.stdout.flush()
                    self.last_printed_sec = remaining

                if elapsed >= self.calibrate_time:
                    print("")
                    self.finish_calibration()
                    return

            imu_msg = Imu()
            imu_msg.header.frame_id = "imu_link"
            imu_msg.header.stamp.sec = int(hw_timestamp)
            imu_msg.header.stamp.nanosec = int((hw_timestamp - int(hw_timestamp)) * 1e9)

            imu_msg.angular_velocity.x = phys_gx
            imu_msg.angular_velocity.y = phys_gy
            imu_msg.angular_velocity.z = phys_gz

            imu_msg.linear_acceleration.x = phys_ax
            imu_msg.linear_acceleration.y = phys_ay
            imu_msg.linear_acceleration.z = phys_az

            imu_msg.orientation_covariance = [
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            imu_msg.angular_velocity_covariance = [
                0.01,
                0.0,
                0.0,
                0.0,
                0.01,
                0.0,
                0.0,
                0.0,
                0.01,
            ]
            imu_msg.linear_acceleration_covariance = [
                0.0003,
                0.0,
                0.0,
                0.0,
                0.0003,
                0.0,
                0.0,
                0.0,
                0.0003,
            ]

            self.imu_pub.publish(imu_msg)

        temp_parts = temp_str.split(",")
        if len(temp_parts) == 2:
            temp_msg = Temperature()
            temp_msg.header.frame_id = "imu_link"
            hw_timestamp = float(temp_parts[0])
            temp_msg.header.stamp.sec = int(hw_timestamp)
            temp_msg.header.stamp.nanosec = int(
                (hw_timestamp - int(hw_timestamp)) * 1e9
            )
            temp_msg.temperature = float(
                (int(temp_parts[1]) * self.temp_scale) + self.temp_offset
            )
            temp_msg.variance = 0.0

            self.temp_pub.publish(temp_msg)

    def stop(self):
        self.node.destroy_node()
        rclpy.shutdown()


class HoloscanIMUApp(Application):
    def __init__(self, args):
        super().__init__()
        self.app_args = args

    def compose(self):
        c = self.kwargs("imu_hardware")

        imu_models_config = self.kwargs("imu_models")
        active_model_config = imu_models_config.get(self.app_args.imu_model)
        is_16607 = self.app_args.imu_model == "ADIS16607-2"

        def get_hw_int(val_phys, scale, is_16607):
            if val_phys == 0.0:
                return 0

            correction_phys = -val_phys

            if is_16607:
                hw_scale = scale * 256.0
                raw = int(round(correction_phys / hw_scale))
                return max(min(raw, 32767), -32768)
            else:
                raw = int(round(correction_phys / scale))
                return max(min(raw, 2147483647), -2147483648)

        hw_gx = get_hw_int(
            self.app_args.calib_gx, active_model_config["gyro_scale"], is_16607
        )
        hw_gy = get_hw_int(
            self.app_args.calib_gy, active_model_config["gyro_scale"], is_16607
        )
        hw_gz = get_hw_int(
            self.app_args.calib_gz, active_model_config["gyro_scale"], is_16607
        )
        hw_ax = get_hw_int(
            self.app_args.calib_ax, active_model_config["accel_scale"], is_16607
        )
        hw_ay = get_hw_int(
            self.app_args.calib_ay, active_model_config["accel_scale"], is_16607
        )
        hw_az = get_hw_int(
            self.app_args.calib_az, active_model_config["accel_scale"], is_16607
        )

        imu_hardware = create_imu_operator(
            self,
            "imu_hardware",
            self.app_args.imu_model,
            c.get("hsb_ip", "192.168.0.2"),
            c.get("sync_freq", 120),
            c.get("output_freq", 240),
            c.get("spi_port", 2),
            c.get("spi_cs", 0),
            c.get("spi_div", 10),
            c.get("spi_cpol", 1),
            c.get("spi_cpha", 1),
            c.get("reset_pin", 9),
            c.get("dr_pin", 8),
            c.get("sync_pin", 13),
            hw_gx,
            hw_gy,
            hw_gz,
            hw_ax,
            hw_ay,
            hw_az,
        )

        throttle_condition = PeriodicCondition(
            self, name="hardware_throttle", recess_period=0.002
        )
        imu_hardware.add_arg(throttle_condition)

        ros2_publisher = NativeRos2PublisherOp(
            self,
            name="ros2_publisher",
            imu_model=self.app_args.imu_model,
            model_config=active_model_config,
            local_gravity=c.get("local_gravity", 9.80665),
            calibrate_time=self.app_args.calibrate,
        )

        self.add_flow(
            imu_hardware,
            ros2_publisher,
            {("imu_data", "imu_raw"), ("temp_data", "temp_raw")},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holoscan ADI Precision IMU ROS 2 Bridge"
    )
    parser.add_argument("--config", default="adi_imu_config.yaml")

    parser.add_argument(
        "--imu_model",
        type=str,
        default="ADIS16505-2",
        choices=["ADIS16505-2", "ADIS16607-2"],
        help="Select the target IMU hardware model",
    )
    parser.add_argument(
        "--calibrate",
        type=float,
        default=0.0,
        help="Run in calibration mode for N seconds, then print offsets and exit.",
    )

    parser.add_argument(
        "--calib_gx", type=float, default=0.0, help="Gyro X bias in rad/s"
    )
    parser.add_argument(
        "--calib_gy", type=float, default=0.0, help="Gyro Y bias in rad/s"
    )
    parser.add_argument(
        "--calib_gz", type=float, default=0.0, help="Gyro Z bias in rad/s"
    )
    parser.add_argument(
        "--calib_ax", type=float, default=0.0, help="Accel X bias in m/s^2"
    )
    parser.add_argument(
        "--calib_ay", type=float, default=0.0, help="Accel Y bias in m/s^2"
    )
    parser.add_argument(
        "--calib_az", type=float, default=0.0, help="Accel Z bias in m/s^2"
    )

    args = parser.parse_args()

    app = HoloscanIMUApp(args)
    app.config(args.config)
    app.run()
