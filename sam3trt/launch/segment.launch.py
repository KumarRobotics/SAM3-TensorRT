from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    publish_compressed = LaunchConfiguration("publish_compressed")

    return LaunchDescription([
        DeclareLaunchArgument("publish_compressed", default_value="false"),
        Node(
            package="sam3trt",
            executable="sam3_node",
            name="sam3",
            output="screen",
            parameters=[{"publish_compressed": publish_compressed}],
            remappings=[
                ("/image", "/image"),
                ("/image/compressed", "/cam_driver/image_raw/compressed")]),
            ],
        )
