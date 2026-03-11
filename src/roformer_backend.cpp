#include "roformer_backend.h"
#include "file_utils.h"
#include "log.h"
#include "ort_provider.h"
#include <cmath>
#include <stdexcept>
#include <vector>

void RoFormerBackend::load(const std::string& model_path, bool use_gpu, const Config& config) {
    config_ = config;

    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool gpu_ok = false;
    if (use_gpu)
        gpu_ok = try_cuda_provider(opts);
    if (!gpu_ok)
        opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    auto data = read_file(model_path);
    session_ = std::make_unique<Ort::Session>(env_, data.data(), data.size(), opts);

    loaded_path_ = model_path;
    loaded_gpu_ = gpu_ok;
    loaded_ = true;
    LOG("roformer loaded path=%s size=%zu provider=%s\n",
        model_path.c_str(), data.size(), gpu_ok ? "cuda" : "cpu");
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
    const int freq_bins = config_.stft.n_fft / 2 + 1;

    // Reflect-pad borders for overlap-add chunking
    int border = C - step;
    int padded_len = total_len;
    Eigen::MatrixXf padded;

    if (total_len > 2 * border && border > 0) {
        padded_len = total_len + 2 * border;
        padded.resize(2, padded_len);

        for (int i = 0; i < border; ++i) {
            padded(0, i) = audio(0, border - i);
            padded(1, i) = audio(1, border - i);
        }
        padded.block(0, border, 2, total_len) = audio;
        for (int i = 0; i < border; ++i) {
            padded(0, border + total_len + i) = audio(0, total_len - 2 - i);
            padded(1, border + total_len + i) = audio(1, total_len - 2 - i);
        }
    } else {
        padded = audio;
        border = 0;
    }

    auto window = make_fade_window(C, fade_size);

    int model_stems = config_.num_stems;
    int total_stems = model_stems + (config_.compute_residual ? 1 : 0);

    Eigen::Tensor3dXf result(model_stems, 2, padded_len);
    result.setZero();
    Eigen::MatrixXf counter(2, padded_len);
    counter.setZero();

    int num_chunks = 0;
    for (int i = 0; i < padded_len; i += step) num_chunks++;

    LOG("roformer inference: chunks=%d chunk_size=%d step=%d stems=%d\n",
        num_chunks, C, step, total_stems);

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
        Eigen::MatrixXf chunk(2, C);
        chunk.setZero();
        for (int ch = 0; ch < 2; ++ch)
            for (int s = 0; s < length; ++s)
                chunk(ch, s) = padded(ch, i + s);

        // Reflect-pad short chunks
        if (length < C && length > C / 2 + 1) {
            for (int ch = 0; ch < 2; ++ch)
                for (int s = length; s < C; ++s) {
                    int reflect_idx = 2 * length - 2 - s;
                    if (reflect_idx >= 0)
                        chunk(ch, s) = chunk(ch, reflect_idx);
                }
        }

        // STFT each channel
        Eigen::VectorXf ch0 = chunk.row(0);
        Eigen::VectorXf ch1 = chunk.row(1);
        auto spec0 = stft(ch0, config_.stft);
        auto spec1 = stft(ch1, config_.stft);
        int time_frames = spec0.dimension(1);

        // Build ONNX input: (batch=1, channels=2, freq_bins, time_frames, 2)
        size_t input_elems = static_cast<size_t>(2) * freq_bins * time_frames * 2;
        std::vector<float> input_data(input_elems);
        for (int ch = 0; ch < 2; ++ch) {
            auto& spec = (ch == 0) ? spec0 : spec1;
            for (int f = 0; f < freq_bins; ++f)
                for (int t = 0; t < time_frames; ++t)
                    for (int c = 0; c < 2; ++c)
                        input_data[((ch * freq_bins + f) * time_frames + t) * 2 + c] = spec(f, t, c);
        }

        std::array<int64_t, 5> input_shape = {1, 2, static_cast<int64_t>(freq_bins), static_cast<int64_t>(time_frames), 2};
        auto input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        // Output: (1, num_stems, 2, freq_bins, time_frames, 2)
        auto& out_tensor = outputs[0];
        auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
        const float* out_data = out_tensor.GetTensorData<float>();

        int out_stems = static_cast<int>(out_shape[1]);
        int out_ch = static_cast<int>(out_shape[2]);
        int out_freq = static_cast<int>(out_shape[3]);
        int out_time = static_cast<int>(out_shape[4]);

        // ISTFT each stem/channel and accumulate
        Eigen::VectorXf win = window;
        if (i == 0)
            for (int w = 0; w < fade_size; ++w) win(w) = 1.0f;
        if (i + C >= padded_len)
            for (int w = 0; w < fade_size; ++w) win(C - 1 - w) = 1.0f;

        for (int st = 0; st < out_stems && st < model_stems; ++st) {
            for (int ch = 0; ch < std::min(out_ch, 2); ++ch) {
                // Extract spectrogram for this stem/channel
                Eigen::Tensor<float, 3> spec(out_freq, out_time, 2);
                for (int f = 0; f < out_freq; ++f)
                    for (int t = 0; t < out_time; ++t)
                        for (int c = 0; c < 2; ++c) {
                            int idx = (((st * out_ch + ch) * out_freq + f) * out_time + t) * 2 + c;
                            spec(f, t, c) = out_data[idx];
                        }

                Eigen::VectorXf reconstructed = istft(spec, config_.stft, C);
                int write_len = std::min(length, static_cast<int>(reconstructed.size()));
                for (int s = 0; s < write_len; ++s)
                    result(st, ch, i + s) += reconstructed(s) * win(s);
            }
        }
        for (int ch = 0; ch < 2; ++ch)
            for (int s = 0; s < length; ++s)
                counter(ch, i + s) += win(s);

        chunk_idx++;
        if (cb) cb(static_cast<float>(chunk_idx) / static_cast<float>(num_chunks), "");
    }

    // Normalize by overlap count
    for (int st = 0; st < model_stems; ++st)
        for (int ch = 0; ch < 2; ++ch)
            for (int s = 0; s < padded_len; ++s)
                if (counter(ch, s) > 0.0f)
                    result(st, ch, s) /= counter(ch, s);

    // Trim padding
    Eigen::Tensor3dXf trimmed(total_stems, 2, total_len);
    trimmed.setZero();
    for (int st = 0; st < model_stems; ++st)
        for (int ch = 0; ch < 2; ++ch)
            for (int s = 0; s < total_len; ++s)
                trimmed(st, ch, s) = result(st, ch, border + s);

    // Compute residual stem if needed
    if (config_.compute_residual) {
        for (int ch = 0; ch < 2; ++ch)
            for (int s = 0; s < total_len; ++s) {
                float sum = 0.0f;
                for (int st = 0; st < model_stems; ++st)
                    sum += trimmed(st, ch, s);
                trimmed(model_stems, ch, s) = audio(ch, s) - sum;
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

        LOG("roformer GPU inference failed: %s\n", e.what());
        LOG("roformer falling back to CPU\n");

        Ort::SessionOptions cpu_opts;
        cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        cpu_opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

        auto data = read_file(loaded_path_);
        session_ = std::make_unique<Ort::Session>(env_, data.data(), data.size(), cpu_opts);
        loaded_gpu_ = false;

        return run_inference(audio, cb);
    }
}
