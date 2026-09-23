#include "ros/Sam3Node.hpp"
#include "sam3trt/Visualize.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace enc = sensor_msgs::image_encodings;

Sam3Node::Sam3Node(const rclcpp::NodeOptions& options) : rclcpp::Node("sam3", options)
{
    const std::string models_path = declare_parameter<std::string>("models_path", defaultModelsPath());
    const std::string config_path = declare_parameter<std::string>("config_path", defaultConfigPath());
    const bool publish_compressed = declare_parameter<bool>("publish_compressed", false);

    model_ = std::make_unique<Sam3Model>(models_path, config_path + "/merges.txt", config_path + "/vocab.json", config_path + "/params.json");

    image_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    classes_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions image_options;
    image_options.callback_group = image_group_;

    rclcpp::SubscriptionOptions classes_options;
    classes_options.callback_group = classes_group_;

    image_sub_ = create_subscription<Image>(
        "/image", rclcpp::SensorDataQoS(),
        [this](const Image::ConstSharedPtr& msg) { onImage(msg); },
        image_options);
    compressed_sub_ = create_subscription<CompressedImage>(
        "/image/compressed", rclcpp::SensorDataQoS(),
        [this](const CompressedImage::ConstSharedPtr& msg) { onCompressedImage(msg); },
        image_options);
    classes_sub_ = create_subscription<String>(
        "/classes", rclcpp::QoS(1).transient_local(),
        [this](const String::ConstSharedPtr& msg) { onClasses(msg); },
        classes_options);

    image_pub_ = create_publisher<Image>("/segmentation", 10);
    if (publish_compressed) {
        compressed_pub_ = create_publisher<CompressedImage>("/segmentation/compressed", 10);
    }
}

std::string Sam3Node::defaultModelsPath() {
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("HOME not set");
    return std::string(home) + "/models";
}

std::string Sam3Node::defaultConfigPath() {
    return ament_index_cpp::get_package_share_directory("sam3trt") + "/config";
}

void Sam3Node::onImage(const Image::ConstSharedPtr& msg) {
    static const std::map<std::string, std::pair<int, int>> formats = {
        {enc::BGR8, {CV_8UC3, -1}},
        {enc::RGB8, {CV_8UC3, cv::COLOR_RGB2BGR}},
        {enc::MONO8, {CV_8UC1, cv::COLOR_GRAY2BGR}},
        {enc::BGRA8, {CV_8UC4, cv::COLOR_BGRA2BGR}},
        {enc::RGBA8, {CV_8UC4, cv::COLOR_RGBA2BGR}}
    };

    const auto format = formats.find(msg->encoding);
    if (format == formats.end()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Unsupported image encoding: %s", msg->encoding.c_str());
        return;
    }

    const auto [type, conversion] = format->second;
    const cv::Mat image(msg->height, msg->width, type, const_cast<uint8_t*>(msg->data.data()), msg->step);
    if (conversion < 0) {
        process(image, msg->header);
        return;
    }

    cv::Mat bgr;
    cv::cvtColor(image, bgr, conversion);
    process(bgr, msg->header);
}

void Sam3Node::onCompressedImage(const CompressedImage::ConstSharedPtr& msg) {
    const cv::Mat image = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (image.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Failed to decode compressed image: %s", msg->format.c_str());
        return;
    }

    process(image, msg->header);
}

std::vector<std::string> Sam3Node::parseClasses(const std::string& text) {
    std::vector<std::string> classes;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const size_t first = token.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        const size_t last = token.find_last_not_of(" \t");
        classes.push_back(token.substr(first, last - first + 1));
    }
    return classes;
}

void Sam3Node::onClasses(const String::ConstSharedPtr& msg) {
    std::vector<std::string> classes = parseClasses(msg->data);
    RCLCPP_INFO(get_logger(), "Segmenting %zu class(es): %s", classes.size(), msg->data.c_str());

    std::lock_guard<std::mutex> lock(classes_mutex_);
    classes_ = std::move(classes);
}

std::vector<std::string> Sam3Node::currentClasses() {
    std::lock_guard<std::mutex> lock(classes_mutex_);
    return classes_;
}

void Sam3Node::process(const cv::Mat& image, const std_msgs::msg::Header& header) {
    const std::vector<std::string> classes = currentClasses();
    if (classes.empty()) return;

    const cv::Mat viz = visualize(image, model_->infer(image, classes));

    auto out = std::make_unique<Image>();
    out->header = header;
    out->height = viz.rows;
    out->width = viz.cols;
    out->encoding = enc::BGR8;
    out->step = static_cast<uint32_t>(viz.step);
    out->data.assign(viz.datastart, viz.dataend);
    image_pub_->publish(std::move(out));

    if (!compressed_pub_) return;

    auto compressed = std::make_unique<CompressedImage>();
    compressed->header = header;
    compressed->format = "jpeg";
    cv::imencode(".jpg", viz, compressed->data);
    compressed_pub_->publish(std::move(compressed));
}

RCLCPP_COMPONENTS_REGISTER_NODE(Sam3Node)
