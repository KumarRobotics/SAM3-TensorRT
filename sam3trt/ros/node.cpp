#include "ros/Sam3Node.hpp"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::executors::MultiThreadedExecutor executor;
    auto node = std::make_shared<Sam3Node>();
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
