#include "sam3tensorrt/Sam3Processor.hpp"

Sam3Processor::Sam3Processor(const std::string& merges_path,const std::string& vocab_path, const Params& params) 
    : params_(params), tokenizer_(merges_path, vocab_path)
{
    cudaStreamCreate(&stream_);
    allocateBuffers();
}

Sam3Processor::Sam3Processor(const std::string& merges_path, const std::string& vocab_path) : Sam3Processor(merges_path, vocab_path, Params{})
{ }

Sam3Processor::~Sam3Processor() 
{
    cudaStreamDestroy(stream_);

    cudaFreeHost(h_bgr_);
    cudaFreeHost(h_original_sizes_);
    cudaFreeHost(h_input_ids_);
    cudaFreeHost(h_attention_mask_);
    cudaFreeHost(h_attention_mask_f_);

    cudaFree(d_bgr_);
    cudaFree(d_image_);
    cudaFree(d_original_sizes_);
    cudaFree(d_input_ids_);
    cudaFree(d_attention_mask_);
    cudaFree(d_attention_mask_f_);
}

void Sam3Processor::allocateBuffers()
{
    const int sz = params_.image_size; // 644
    const int img_px = sz * sz;
    const int txt_len = params_.text_padding; // 32

    cudaMallocHost(&h_bgr_, img_px * 3 * sizeof(uint8_t));
    cudaMallocHost(&h_original_sizes_, 2 * sizeof(int32_t));

    cudaMalloc(&d_bgr_, img_px * 3 * sizeof(uint8_t));
    cudaMalloc(&d_image_, 3 * img_px * sizeof(float));
    cudaMalloc(&d_original_sizes_, 2 * sizeof(int32_t));

    cudaMallocHost(&h_input_ids_, txt_len * sizeof(int64_t));
    cudaMallocHost(&h_attention_mask_, txt_len * sizeof(uint8_t));
    cudaMallocHost(&h_attention_mask_f_, txt_len * sizeof(__half));
 
    cudaMalloc(&d_input_ids_, txt_len * sizeof(int64_t));
    cudaMalloc(&d_attention_mask_, txt_len * sizeof(bool));
    cudaMalloc(&d_attention_mask_f_, txt_len * sizeof(__half));
}

void Sam3Processor::uploadImage(const cv::Mat& resized)
{
    const int sz = params_.image_size;
    const size_t n = static_cast<size_t>(sz * sz * 3);

    std::memcpy(h_bgr_, resized.data, n);
    cudaMemcpyAsync(d_bgr_, h_bgr_, n, cudaMemcpyHostToDevice, stream_);
}

void Sam3Processor::uploadOriginalSizes(int h, int w)
{
    h_original_sizes_[0] = static_cast<int32_t>(h);
    h_original_sizes_[1] = static_cast<int32_t>(w);
    cudaMemcpyAsync(d_original_sizes_, h_original_sizes_, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
}

void Sam3Processor::uploadText(const std::vector<int64_t>& ids, const std::vector<uint8_t>& mask, const std::vector<__half>& mask_f) 
{
    const int len = params_.text_padding;

    std::memcpy(h_input_ids_, ids.data(),  len * sizeof(int64_t));
    std::memcpy(h_attention_mask_, mask.data(), len * sizeof(uint8_t));
    std::memcpy(h_attention_mask_f_, mask_f.data(), len * sizeof(__half)); 

    cudaMemcpyAsync(d_input_ids_, h_input_ids_, len * sizeof(int64_t), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_attention_mask_, h_attention_mask_, len * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_attention_mask_f_, h_attention_mask_f_, len * sizeof(__half), cudaMemcpyHostToDevice, stream_);
}

Sam3ImageInput Sam3Processor::preprocessImage(const cv::Mat& image)
{
    if (image.empty()) {
        throw std::runtime_error("Sam3Processor::preprocessImage: empty image");
    }
    if (image.type() != CV_8UC3) {
        throw std::runtime_error("Sam3Processor::preprocessImage: expected CV_8UC3 BGR image");
    }

    const int sz = params_.image_size;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(sz, sz), 0, 0, cv::INTER_LINEAR);

    uploadOriginalSizes(image.rows, image.cols);
    uploadImage(resized);
    normalizeImage(d_bgr_, d_image_, sz, sz, params_.image_mean, params_.image_std, stream_);

    cudaStreamSynchronize(stream_);

    return {d_image_, d_original_sizes_};
}

Sam3TextInput Sam3Processor::preprocessText(const std::string& text) 
{
    if (text.empty()) {
        throw std::runtime_error("Sam3Processor::preprocessText: empty string");
    }

    const int len = params_.text_padding;

    std::vector<int> raw = tokenizer_.tokenize(text);

    const int real_len = std::min(static_cast<int>(raw.size()), len);

    std::vector<int64_t> ids(len, 0);
    std::vector<uint8_t> mask(len, 1);
    std::vector<__half> mask_f(len, __float2half(1.0f));

    for (int i = 0; i < real_len; ++i) {
        ids[i] = static_cast<int64_t>(raw[i]);
        mask[i] = 0;
        mask_f[i] = __float2half(0.0f);
    }

    uploadText(ids, mask, mask_f);

    cudaStreamSynchronize(stream_);

    return { d_input_ids_, d_attention_mask_, d_attention_mask_f_ };
}

Sam3Input Sam3Processor::preprocess(const cv::Mat& image, const std::vector<std::string>& texts)
{
    Sam3ImageInput image_input = preprocessImage(image);

    std::vector<Sam3TextInput> text_inputs;
    for (const std::string& t : texts) {
        text_inputs.push_back(preprocessText(t));
    }

    return {image_input, text_inputs};
}
