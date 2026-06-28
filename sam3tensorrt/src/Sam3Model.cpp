#include "sam3tensorrt/Sam3Model.hpp"

#include <stdexcept>

Sam3Model::Sam3Model(const std::string& model_path) : 
    image_encoder_(model_path+"/image_encoder.plan"), text_encoder_(model_path+"/text_encoder.plan"), mask_decoder_(model_path+"/mask_decoder.plan")
{
    cudaStreamCreate(&img_stream_);
    cudaStreamCreate(&txt_stream_);
    cudaStreamCreate(&decoder_stream_);
}

Sam3Model::~Sam3Model() 
{
    cudaStreamDestroy(img_stream_);
    cudaStreamDestroy(txt_stream_);
    cudaStreamDestroy(decoder_stream_);
}

Sam3DecoderOutput Sam3Model::forward(const float* d_image, const int32_t* d_input_ids, const int32_t* d_attention_mask, const int32_t* d_original_sizes)
{
    Sam3ImageFeatures image_features = image_encoder_.encode(d_image, img_stream_);
    Sam3TextFeatures  text_features  = text_encoder_.encode(d_input_ids, d_attention_mask, txt_stream_);

    cudaStreamSynchronize(img_stream_);
    cudaStreamSynchronize(txt_stream_);

    Sam3DecoderOutput output = mask_decoder_.decode(image_features, text_features, d_attention_mask, d_original_sizes, decoder_stream_);

    cudaStreamSynchronize(decoder_stream_);
    return output;
}
