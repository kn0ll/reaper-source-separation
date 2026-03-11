#include "demucs_backend.h"
#include "log.h"
#include "ort_provider.h"
#include <fstream>
#include <stdexcept>

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
    LOG("demucs loaded path=%s size=%zu provider=%s\n",
        model_path.c_str(), data.size(), gpu_ok ? "cuda" : "cpu");
}

Eigen::Tensor3dXf DemucsBackend::infer(const Eigen::MatrixXf& audio, ProgressCallback cb) {
    if (!loaded_)
        throw std::runtime_error("DemucsBackend: model not loaded");

    try {
        return demucsonnx::demucs_inference(model_, audio, cb);
    } catch (const std::exception& e) {
        if (!loaded_gpu_) throw;

        LOG("demucs GPU inference failed: %s\n", e.what());
        LOG("demucs falling back to CPU\n");

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
