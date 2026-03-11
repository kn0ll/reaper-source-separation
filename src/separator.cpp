#include "separator.h"
#include "audio_io.h"
#include "backend.h"
#include "demucs_backend.h"
#include "log.h"
#include "roformer_backend.h"
#include "model_manager.h"
#include <filesystem>
#include <cmath>
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

static std::mutex                    g_backend_mutex;
static std::unique_ptr<DemucsBackend>    g_demucs;
static std::unique_ptr<RoFormerBackend>  g_roformer;

static fs::path temp_parent_dir() {
    return fs::temp_directory_path() / "reaper_stem_separation_plugin";
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

static void worker(SeparationRequest req) {
    try {
        g_progress.store(0.0f);

        auto* info = model_manager::find_model(req.model_id);
        if (!info)
            throw std::runtime_error("Unknown model: " + req.model_id);

        bool use_gpu = true;

        LOG("worker: model=%s backend=%s source=%s\n",
            req.model_id.c_str(),
            info->backend == model_manager::BackendType::Demucs ? "demucs" : "roformer",
            req.source_path.c_str());

        set_status("Loading model...");

        Eigen::Tensor3dXf targets;
        std::vector<std::string> stem_names = info->stem_names;
        int sample_rate = info->sample_rate;

        {
            std::lock_guard<std::mutex> lk(g_backend_mutex);

            if (info->backend == model_manager::BackendType::Demucs) {
                if (!g_demucs) g_demucs = std::make_unique<DemucsBackend>();
                if (!g_demucs->is_loaded() || g_demucs->loaded_path() != req.model_path
                    || g_demucs->loaded_with_gpu() != use_gpu) {
                    LOG("loading demucs from %s\n", req.model_path.c_str());
                    g_demucs->load(req.model_path, use_gpu);
                }
            } else {
                if (!g_roformer) g_roformer = std::make_unique<RoFormerBackend>();
                RoFormerBackend::Config cfg;
                cfg.chunk_size = info->chunk_size;
                cfg.num_overlap = info->num_overlap;
                cfg.num_stems = static_cast<int>(info->stem_names.size())
                                - (info->compute_residual ? 1 : 0);
                cfg.compute_residual = info->compute_residual;
                cfg.stft = {info->stft_n_fft, info->stft_hop_length, info->stft_win_length};
                LOG("roformer config: chunk=%d overlap=%d stems=%d stft_nfft=%d stft_hop=%d stft_win=%d\n",
                    cfg.chunk_size, cfg.num_overlap, cfg.num_stems,
                    cfg.stft.n_fft, cfg.stft.hop_length, cfg.stft.win_length);
                if (!g_roformer->is_loaded() || g_roformer->loaded_path() != req.model_path
                    || g_roformer->loaded_with_gpu() != use_gpu) {
                    LOG("loading roformer from %s\n", req.model_path.c_str());
                    g_roformer->load(req.model_path, use_gpu, cfg);
                }
            }
        }

        if (g_cancel.load()) throw std::runtime_error("Cancelled");

        g_progress.store(0.05f);
        set_status("Reading audio...");

        LOG("reading audio from %s\n", req.source_path.c_str());
        Eigen::MatrixXf audio = audio_io::load(req.source_path, sample_rate);
        LOG("audio loaded: channels=%d samples=%d duration=%.1fs\n",
            (int)audio.rows(), (int)audio.cols(),
            (float)audio.cols() / (float)sample_rate);

        float duration = static_cast<float>(audio.cols()) / static_cast<float>(sample_rate);
        g_progress.store(0.10f);
        static constexpr int kStatusBufLen = 128;
        char dur_buf[kStatusBufLen];
        snprintf(dur_buf, sizeof(dur_buf), "Separating %.1fs of audio...", duration);
        set_status(dur_buf);

        if (g_cancel.load()) throw std::runtime_error("Cancelled");

        int seg_count = 0;
        int total_segs = 0;

        ProgressCallback cb = [&](float p, const std::string&) {
            if (g_cancel.load()) throw std::runtime_error("Cancelled");
            ++seg_count;

            if (total_segs == 0 && p > 0.0f)
                total_segs = static_cast<int>(std::round(1.0f / p));

            g_progress.store(0.10f + p * 0.80f);

            if (total_segs > 1)
                set_status("Separating segment " + std::to_string(seg_count) + "/" + std::to_string(total_segs) + "...");
            else
                set_status("Separating...");
        };

        LOG("starting inference\n");
        {
            std::lock_guard<std::mutex> lk(g_backend_mutex);
            if (info->backend == model_manager::BackendType::Demucs)
                targets = g_demucs->infer(audio, cb);
            else
                targets = g_roformer->infer(audio, cb);
        }
        LOG("inference done: stems=%d channels=%d samples=%d\n",
            (int)targets.dimension(0), (int)targets.dimension(1), (int)targets.dimension(2));

        if (g_cancel.load()) throw std::runtime_error("Cancelled");

        g_progress.store(0.90f);
        set_status("Writing stem files...");

        std::string out_dir = make_output_dir();
        LOG("writing stems to %s\n", out_dir.c_str());
        auto stems = audio_io::write_stems(targets, stem_names, sample_rate, out_dir);

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
        if (std::string(e.what()) != "Cancelled")
            LOG("worker error: %s\n", e.what());
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
    std::lock_guard<std::mutex> lk(g_backend_mutex);
    g_demucs.reset();
    g_roformer.reset();
}

void separator::cleanup_temp_files() {
    std::error_code ec;
    fs::remove_all(temp_parent_dir(), ec);
}
