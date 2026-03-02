#include "separator.h"
#include "audio_io.h"
#include "demucs.hpp"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace fs = std::filesystem;

static std::atomic<separator::State> g_state{separator::State::Idle};
static std::atomic<float>            g_progress{0.0f};
static std::atomic<bool>             g_cancel{false};

static std::mutex                    g_mutex;
static std::string                   g_status;
static std::string                   g_error;
static SeparationResult              g_result;

static std::thread                   g_thread;

static std::mutex                    g_model_mutex;
static demucsonnx::demucs_model      g_model;
static std::string                   g_loaded_model_path;
static bool                          g_loaded_with_gpu = false;

static fs::path temp_parent_dir() {
    return fs::temp_directory_path() / "reaper_source_separation";
}

static std::string make_output_dir() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    auto subdir = temp_parent_dir() / std::to_string(dist(gen));
    fs::create_directories(subdir);
    return subdir.string();
}

static void set_status(const std::string& msg) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_status = msg;
}

static bool try_cuda_provider(Ort::SessionOptions& opts) {
    auto providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end())
        return false;
    try {
        OrtCUDAProviderOptions cuda_opts{};
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        return true;
    } catch (...) {
        return false;
    }
}

static void worker(SeparationRequest req) {
    try {
        g_progress.store(0.0f);

        bool using_gpu = false;
        {
            std::lock_guard<std::mutex> lk(g_model_mutex);

            Ort::SessionOptions opts;
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            using_gpu = try_cuda_provider(opts);
            if (!using_gpu)
                opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

            set_status(using_gpu ? "Loading model (GPU)..." : "Loading model...");

            if (g_loaded_model_path != req.model_path || g_loaded_with_gpu != using_gpu) {
                std::ifstream f(req.model_path, std::ios::binary | std::ios::ate);
                if (!f) throw std::runtime_error("Cannot open model: " + req.model_path);

                std::streamsize sz = f.tellg();
                f.seekg(0, std::ios::beg);
                std::vector<char> data(sz);
                if (!f.read(data.data(), sz))
                    throw std::runtime_error("Failed to read model file");

                g_model = demucsonnx::demucs_model{};
                if (!demucsonnx::load_model(data, g_model, opts))
                    throw std::runtime_error("Failed to load ONNX model");

                g_loaded_model_path = req.model_path;
                g_loaded_with_gpu = using_gpu;
                fprintf(stderr, "[reaper-source-separation] using %s for inference\n",
                        using_gpu ? "GPU (CUDA)" : "CPU");
            }
        }

        if (g_cancel.load()) throw std::runtime_error("Cancelled");

        g_progress.store(0.05f);
        set_status("Reading audio...");

        Eigen::MatrixXf audio = audio_io::load(req.source_path);
        float duration = static_cast<float>(audio.cols()) / demucsonnx::SUPPORTED_SAMPLE_RATE;
        g_progress.store(0.10f);
        char dur_buf[128];
        snprintf(dur_buf, sizeof(dur_buf), "Separating %.1fs of audio%s...",
                 duration, using_gpu ? " (GPU)" : "");
        set_status(dur_buf);

        if (g_cancel.load()) throw std::runtime_error("Cancelled");

        int seg_count = 0;
        float prev_p = -1.0f;
        int total_segs = 0;

        demucsonnx::ProgressCallback cb = [&](float p, const std::string&) {
            if (g_cancel.load()) throw std::runtime_error("Cancelled");
            ++seg_count;

            // Estimate total segments from the first callback's progress increment
            if (total_segs == 0 && p > 0.0f)
                total_segs = static_cast<int>(std::round(1.0f / p));

            g_progress.store(0.10f + p * 0.80f);

            if (total_segs > 1)
                set_status("Separating segment " + std::to_string(seg_count) + "/" + std::to_string(total_segs) + "...");
            else
                set_status("Separating...");

            prev_p = p;
        };

        Eigen::Tensor3dXf targets;
        {
            std::lock_guard<std::mutex> lk(g_model_mutex);
            targets = demucsonnx::demucs_inference(g_model, audio, cb);
        }

        if (g_cancel.load()) throw std::runtime_error("Cancelled");

        g_progress.store(0.90f);
        set_status("Writing stem files...");

        std::string out_dir = make_output_dir();
        auto stems = audio_io::write_stems(targets, g_model.nb_sources, out_dir);

        SeparationResult res;
        res.request = req;
        res.stems = std::move(stems);

        {
            std::lock_guard<std::mutex> lk(g_mutex);
            g_result = std::move(res);
        }

        g_progress.store(1.0f);
        set_status("Done");
        g_state.store(separator::State::Done);

    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_error = e.what();
        g_state.store(separator::State::Error);
    }
}

void separator::start(const SeparationRequest& req) {
    if (g_state.load() == State::Running) return;

    g_state.store(State::Running);
    g_cancel.store(false);
    g_progress.store(0.0f);
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_error.clear();
        g_status = "Starting...";
        g_result = {};
    }

    if (g_thread.joinable()) g_thread.join();
    g_thread = std::thread(worker, req);
}

void separator::cancel() {
    g_cancel.store(true);
}

separator::State separator::state() {
    return g_state.load();
}

float separator::progress() {
    return g_progress.load();
}

std::string separator::status_message() {
    std::lock_guard<std::mutex> lk(g_mutex);
    return g_status;
}

SeparationResult separator::result() {
    std::lock_guard<std::mutex> lk(g_mutex);
    return g_result;
}

std::string separator::error_message() {
    std::lock_guard<std::mutex> lk(g_mutex);
    return g_error;
}

void separator::reset() {
    if (g_thread.joinable()) g_thread.join();
    g_state.store(State::Idle);
    g_progress.store(0.0f);
    g_cancel.store(false);
    std::lock_guard<std::mutex> lk(g_mutex);
    g_status.clear();
    g_error.clear();
    g_result = {};
}

void separator::cleanup_model() {
    std::lock_guard<std::mutex> lk(g_model_mutex);
    g_model = demucsonnx::demucs_model{};
    g_loaded_model_path.clear();
    g_loaded_with_gpu = false;
}

void separator::cleanup_temp_files() {
    std::error_code ec;
    fs::remove_all(temp_parent_dir(), ec);
}
