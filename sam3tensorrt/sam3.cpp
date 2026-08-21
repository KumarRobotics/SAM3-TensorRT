#include "sam3tensorrt/Sam3Model.hpp"
#include "sam3tensorrt/Visualize.hpp"

static const std::string TEST_IMAGE = TEST_IMAGE_PATH;
static const std::string CONFIG_DIR = SAM3_CONFIG_DIR;

static std::string modelsDir() {
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("HOME not set");
    return std::string(home) + "/models";
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <label>\n";
        return 1;
    }
    const std::string label = argv[1];

    const std::string models = modelsDir();
    Sam3Model model(models, CONFIG_DIR + "/merges.txt", CONFIG_DIR + "/vocab.json", CONFIG_DIR + "/params.json");

    cv::Mat image = cv::imread(TEST_IMAGE);
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    Sam3Result result = model.infer(rgb, label);
    std::cout << "[Sam3] Detetected: " << result.detections.size() << " object(s)\n";

    cv::Mat viz = visualize(image, result);
    cv::imwrite("../test_viz.png", viz); 

    return 0;
}
