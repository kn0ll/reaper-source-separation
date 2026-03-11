#include "roformer_backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

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

void RoFormerBackend::load(const std::string& model_path, bool use_gpu, const Config& config) {
    config_ = config;

    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool gpu_ok = false;
    if (use_gpu)
        gpu_ok = try_cuda_provider(opts);
    if (!gpu_ok)
        opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    std::ifstream f(model_path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open model: " + model_path);
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data(sz);
    if (!f.read(data.data(), sz))
        throw std::runtime_error("Failed to read model file");

    session_ = std::make_unique<Ort::Session>(env_, data.data(), data.size(), opts);

    loaded_path_ = model_path;
    loaded_gpu_ = gpu_ok;
    loaded_ = true;
    fprintf(stderr, "[reaper-stem-separation-plugin] roformer using %s\n",
            gpu_ok ? "GPU (CUDA)" : "CPU");
}

static Eigen::VectorXf make_fade_window(int size, int fade_size) {
    Eigen::VectorXf window = Eigen::VectorXf::Ones(size);
    for (int i = 0; i < fade_size; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(fade_size);
        window(i) = t;
        window(size - 1 - i) = t;
    }
    return window;
}

Eigen::Tensor3dXf RoFormerBackend::run_inference(const Eigen::MatrixXf& audio, ProgressCallback cb) {
    const int C = config_.chunk_size;
    const int N = config_.num_overlap;
    const int step = C / N;
    const int fade_size = C / 10;
    const int total_len = static_cast<int>(audio.cols());

    // Reflect-pad borders
    int border = C - step;
    int padded_len = total_len;
    Eigen::MatrixXf padded;

    if (total_len > 2 * border && border > 0) {
        padded_len = total_len + 2 * border;
        padded.resize(2, padded_len);

        // Left reflection
        for (int i = 0; i < border; ++i) {
            padded(0, i) = audio(0, border - i);
            padded(1, i) = audio(1, border - i);
        }
        // Center (original)
        padded.block(0, border, 2, total_len) = audio;
        // Right reflection
        for (int i = 0; i < border; ++i) {
            padded(0, border + total_len + i) = audio(0, total_len - 2 - i);
            padded(1, border + total_len + i) = audio(1, total_len - 2 - i);
        }
    } else {
        padded = audio;
        border = 0;
    }

    auto window = make_fade_window(C, fade_size);

    // Determine output stem count from ONNX model
    auto output_info = session_->GetOutputTypeInfo(0);
    auto shape = output_info.GetTensorTypeAndShapeInfo().GetShape();
    // Expected output: (1, num_stems, 2, chunk_samples) or (num_stems, 2, chunk_samples)
    int model_stems = config_.num_stems;
    if (shape.size() == 4) model_stems = static_cast<int>(shape[1]);
    else if (shape.size() == 3) model_stems = static_cast<int>(shape[0]);

    int total_stems = model_stems + (config_.compute_residual ? 1 : 0);

    // Accumulators
    Eigen::Tensor3dXf result(model_stems, 2, padded_len);
    result.setZero();
    Eigen::MatrixXf counter(2, padded_len);
    counter.setZero();

    int num_chunks = 0;
    for (int i = 0; i < padded_len; i += step) num_chunks++;

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    auto output_name = session_->GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    int chunk_idx = 0;
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (int i = 0; i < padded_len; i += step) {
        int length = std::min(C, padded_len - i);

        // Extract chunk, zero-pad if shorter than C
        std::vector<float> chunk_data(2 * C, 0.0f);
        for (int ch = 0; ch < 2; ++ch) {
            for (int s = 0; s < length; ++s) {
                chunk_data[ch * C + s] = padded(ch, i + s);
            }
        }

        // Reflect-pad short chunks
        if (length < C && length > C / 2 + 1) {
            for (int ch = 0; ch < 2; ++ch) {
                for (int s = length; s < C; ++s) {
                    int reflect_idx = 2 * length - 2 - s;
                    if (reflect_idx >= 0)
                        chunk_data[ch * C + s] = chunk_data[ch * C + reflect_idx];
                }
            }
        }

        // Create ORT input tensor: (1, 2, C)
        std::array<int64_t, 3> input_shape = {1, 2, static_cast<int64_t>(C)};
        auto input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, chunk_data.data(), chunk_data.size(), input_shape.data(), input_shape.size());

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        auto& out_tensor = outputs[0];
        auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
        const float* out_data = out_tensor.GetTensorData<float>();

        // Parse output shape: (1, stems, 2, C) or (stems, 2, C)
        int out_stems, out_channels, out_samples;
        if (out_shape.size() == 4) {
            out_stems = static_cast<int>(out_shape[1]);
            out_channels = static_cast<int>(out_shape[2]);
            out_samples = static_cast<int>(out_shape[3]);
        } else {
            out_stems = static_cast<int>(out_shape[0]);
            out_channels = static_cast<int>(out_shape[1]);
            out_samples = static_cast<int>(out_shape[2]);
        }

        // Apply window and accumulate
        Eigen::VectorXf win = window;
        if (i == 0) {
            for (int w = 0; w < fade_size; ++w) win(w) = 1.0f;
        }
        if (i + C >= padded_len) {
            for (int w = 0; w < fade_size; ++w) win(C - 1 - w) = 1.0f;
        }

        int write_len = std::min(length, out_samples);
        for (int st = 0; st < out_stems && st < model_stems; ++st) {
            for (int ch = 0; ch < std::min(out_channels, 2); ++ch) {
                for (int s = 0; s < write_len; ++s) {
                    float val = out_data[st * out_channels * out_samples + ch * out_samples + s];
                    result(st, ch, i + s) += val * win(s);
                }
            }
        }
        for (int ch = 0; ch < 2; ++ch) {
            for (int s = 0; s < write_len; ++s) {
                counter(ch, i + s) += win(s);
            }
        }

        chunk_idx++;
        if (cb) cb(static_cast<float>(chunk_idx) / static_cast<float>(num_chunks), "");
    }

    // Normalize by overlap count
    for (int st = 0; st < model_stems; ++st) {
        for (int ch = 0; ch < 2; ++ch) {
            for (int s = 0; s < padded_len; ++s) {
                if (counter(ch, s) > 0.0f)
                    result(st, ch, s) /= counter(ch, s);
            }
        }
    }

    // Trim padding
    Eigen::Tensor3dXf trimmed(total_stems, 2, total_len);
    trimmed.setZero();
    for (int st = 0; st < model_stems; ++st) {
        for (int ch = 0; ch < 2; ++ch) {
            for (int s = 0; s < total_len; ++s) {
                trimmed(st, ch, s) = result(st, ch, border + s);
            }
        }
    }

    // Compute residual stem if needed (instrumental = mix - sum_of_model_stems)
    if (config_.compute_residual) {
        for (int ch = 0; ch < 2; ++ch) {
            for (int s = 0; s < total_len; ++s) {
                float sum = 0.0f;
                for (int st = 0; st < model_stems; ++st)
                    sum += trimmed(st, ch, s);
                trimmed(model_stems, ch, s) = audio(ch, s) - sum;
            }
        }
    }

    return trimmed;
}

Eigen::Tensor3dXf RoFormerBackend::infer(const Eigen::MatrixXf& audio, ProgressCallback cb) {
    if (!loaded_)
        throw std::runtime_error("RoFormerBackend: model not loaded");

    try {
        return run_inference(audio, cb);
    } catch (const std::exception& e) {
        if (!loaded_gpu_) throw;

        fprintf(stderr, "[reaper-stem-separation-plugin] GPU inference failed: %s\n", e.what());
        fprintf(stderr, "[reaper-stem-separation-plugin] falling back to CPU\n");

        // Reload with CPU
        Ort::SessionOptions cpu_opts;
        cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        cpu_opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

        std::ifstream f(loaded_path_, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("Cannot open model: " + loaded_path_);
        std::streamsize sz = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> data(sz);
        if (!f.read(data.data(), sz))
            throw std::runtime_error("Failed to read model file");

        session_ = std::make_unique<Ort::Session>(env_, data.data(), data.size(), cpu_opts);
        loaded_gpu_ = false;

        return run_inference(audio, cb);
    }
}
