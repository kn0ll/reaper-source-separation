#include "demucs_backend.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

static bool try_cuda_provider(Ort::SessionOptions& opts) {
    auto providers = Ort::GetAvailableProviders();
    fprintf(stderr, "[reaper-stem-separation-plugin] available providers:");
    for (const auto& p : providers) fprintf(stderr, " %s", p.c_str());
    fprintf(stderr, "\n");

    if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end())
        return false;
    try {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        fprintf(stderr, "[reaper-stem-separation-plugin] CUDA provider attached\n");
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[reaper-stem-separation-plugin] CUDA provider failed: %s\n", e.what());
        return false;
    } catch (...) {
        fprintf(stderr, "[reaper-stem-separation-plugin] CUDA provider failed (unknown error)\n");
        return false;
    }
}

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open model: " + path);
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data(sz);
    if (!f.read(data.data(), sz))
        throw std::runtime_error("Failed to read model file");
    return data;
}

void DemucsBackend::load(const std::string& model_path, bool use_gpu) {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool gpu_ok = false;
    if (use_gpu)
        gpu_ok = try_cuda_provider(opts);
    if (!gpu_ok)
        opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    auto data = read_file(model_path);
    model_ = demucsonnx::demucs_model{};
    if (!demucsonnx::load_model(data, model_, opts))
        throw std::runtime_error("Failed to load ONNX model");

    loaded_path_ = model_path;
    loaded_gpu_ = gpu_ok;
    loaded_ = true;
    fprintf(stderr, "[reaper-stem-separation-plugin] demucs using %s\n",
            gpu_ok ? "GPU (CUDA)" : "CPU");
}

Eigen::Tensor3dXf DemucsBackend::infer(const Eigen::MatrixXf& audio, ProgressCallback cb) {
    if (!loaded_)
        throw std::runtime_error("DemucsBackend: model not loaded");

    try {
        return demucsonnx::demucs_inference(model_, audio, cb);
    } catch (const std::exception& e) {
        if (!loaded_gpu_) throw;

        fprintf(stderr, "[reaper-stem-separation-plugin] GPU inference failed: %s\n", e.what());
        fprintf(stderr, "[reaper-stem-separation-plugin] falling back to CPU\n");

        Ort::SessionOptions cpu_opts;
        cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        cpu_opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

        auto data = read_file(loaded_path_);
        model_ = demucsonnx::demucs_model{};
        if (!demucsonnx::load_model(data, model_, cpu_opts))
            throw std::runtime_error("Failed to load ONNX model on CPU");
        loaded_gpu_ = false;

        return demucsonnx::demucs_inference(model_, audio, cb);
    }
}
