#pragma once

#include "sam3trt/Sam3Model.hpp"

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

class Sam3Node : public rclcpp::Node
{
    public:
        explicit Sam3Node(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    private:
        using Image = sensor_msgs::msg::Image;
        using CompressedImage = sensor_msgs::msg::CompressedImage;
        using String = std_msgs::msg::String;

        static std::string defaultModelsPath();
        static std::string defaultConfigPath();

        void onImage(const Image::ConstSharedPtr& msg);
        void onCompressedImage(const CompressedImage::ConstSharedPtr& msg);
        void onClasses(const String::ConstSharedPtr& msg);

        void process(const cv::Mat& image, const std_msgs::msg::Header& header);
        std::vector<std::string> currentClasses();

        static std::vector<std::string> parseClasses(const std::string& text);

        std::unique_ptr<Sam3Model> model_;

        std::mutex classes_mutex_;
        std::vector<std::string> classes_;

        rclcpp::CallbackGroup::SharedPtr image_group_;
        rclcpp::CallbackGroup::SharedPtr classes_group_;

        rclcpp::Subscription<Image>::SharedPtr image_sub_;
        rclcpp::Subscription<CompressedImage>::SharedPtr compressed_sub_;
        rclcpp::Subscription<String>::SharedPtr classes_sub_;
        rclcpp::Publisher<Image>::SharedPtr image_pub_;
        rclcpp::Publisher<CompressedImage>::SharedPtr compressed_pub_;
};
