from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch Alpamayo node that listens to live camera + odometry topics."""
    default_camera_topics = [
        "/sensing/camera/camera3/image_raw/compressed",
        "/sensing/camera/camera1/image_raw/compressed",
        "/sensing/camera/camera4/image_raw/compressed",
        "/sensing/camera/camera2/image_raw/compressed",
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("expert_onnx_path", default_value=""),
            DeclareLaunchArgument("num_diffusion_steps", default_value="10"),
            DeclareLaunchArgument("use_greedy_decode", default_value="false"),
            Node(
                package="alpamayo_ros",
                executable="alpamayo_node",
                name="alpamayo_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "camera_topics": default_camera_topics,
                        "odometry_topic": "/localization/kinematic_state",
                        "trajectory_topic": "/alpamayo/predicted_trajectory",
                        "cot_topic": "/alpamayo/reasoning",
                        "inference_period_sec": 1.0,
                        "expert_onnx_path": LaunchConfiguration("expert_onnx_path"),
                        "num_diffusion_steps": LaunchConfiguration("num_diffusion_steps"),
                        "use_greedy_decode": LaunchConfiguration("use_greedy_decode"),
                    }
                ],
            ),
        ]
    )
