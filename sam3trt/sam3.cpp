#include "sam3trt/Sam3Model.hpp"
#include "sam3trt/Visualize.hpp"

#include <filesystem>

static std::string configDir() {
    const std::filesystem::path exe = std::filesystem::canonical("/proc/self/exe");
    const std::filesystem::path installed = exe.parent_path().parent_path() / SAM3_INSTALL_CONFIG_DIR;
    return std::filesystem::exists(installed) ? installed.string() : SAM3_CONFIG_DIR;
}

static std::string modelsDir() {
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("HOME not set");
    return std::string(home) + "/models";
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <image_path> <label>\n";
        return 1;
    }
    const std::filesystem::path image_path = argv[1];
    const std::string label = argv[2];

    cv::Mat image = cv::imread(image_path.string());
    if (image.empty()) {
        std::cerr << "could not read image: " << image_path.string() << "\n";
        return 1;
    }

    const std::string models = modelsDir();
    const std::string config = configDir();
    Sam3Model model(models, config + "/merges.txt", config + "/vocab.json", config + "/params.json");

    Sam3Result result = model.infer(image, label);
    std::cout << "[Sam3] Detected: " << result.detections.size() << " object(s)\n";

    const std::filesystem::path viz_path = image_path.parent_path() / (image_path.stem().string() + "_viz.png");
    cv::imwrite(viz_path.string(), visualize(image, result));
    std::cout << "[Sam3] Saved: " << viz_path.string() << "\n";

    return 0;
}
