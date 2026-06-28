#include "sam3tensorrt/Sam3ModelBase.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

void Sam3ModelBase::Logger::log(Severity severity, const char* msg) noexcept 
{
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << "\n";
    }
}

Sam3ModelBase::Sam3ModelBase(const std::string& plan_path) 
{
    loadPlan(plan_path);
}

void Sam3ModelBase::loadPlan(const std::string& path) 
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Sam3ModelBase: cannot open plan: " + path);
    }

    const std::streamsize size = file.tellg();
    file.seekg(0);
    std::vector<char> buffer(static_cast<size_t>(size));
    file.read(buffer.data(), size);

    initLibNvInferPlugins(&logger_, "");

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!engine_) {
        throw std::runtime_error("Sam3ModelBase: failed to deserialize engine: " + path);
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Sam3ModelBase: failed to create execution context");
    }
}
