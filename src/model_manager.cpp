#include "model_manager.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;

#ifndef DOWNLOAD_REPO
#define DOWNLOAD_REPO "kn0ll/reaper-stem-separation-plugin"
#endif

#ifndef DOWNLOAD_TAG
#define DOWNLOAD_TAG "latest"
#endif

static const std::vector<model_manager::ModelInfo> g_known_models = {
    {
        "bs_roformer_vocals",
        "bs_roformer_vocals.onnx",
        "Vocals (Best quality)",
        "Isolate vocals with the highest possible clarity",
        {"vocals", "instrumental"},
        model_manager::BackendType::RoFormer,
        44100, 960000, 4, true,
        900'000'000
    },
    {
        "melband_roformer_vocals",
        "melband_roformer_vocals.onnx",
        "Vocals (Fast)",
        "Quick vocal isolation, great for previewing",
        {"vocals", "instrumental"},
        model_manager::BackendType::RoFormer,
        44100, 352800, 2, true,
        900'000'000
    },
    {
        "htdemucs_ft",
        "htdemucs_ft.ort",
        "All Stems",
        "Split a track into its 4 core parts",
        {"drums", "bass", "other", "vocals"},
        model_manager::BackendType::Demucs,
        44100, 0, 0, false,
        210'000'000
    },
    {
        "htdemucs_6s",
        "htdemucs_6s.ort",
        "All Stems + Guitar & Piano",
        "Split a track into 6 parts including guitar and piano",
        {"drums", "bass", "other", "vocals", "guitar", "piano"},
        model_manager::BackendType::Demucs,
        44100, 0, 0, false,
        144'000'000
    },
};

static std::string g_cache_dir;
static std::string g_local_dir;

static std::atomic<model_manager::DownloadState> g_dl_state{model_manager::DownloadState::Idle};
static std::atomic<float>   g_dl_progress{0.0f};
static std::atomic<bool>    g_dl_cancel{false};
static std::mutex           g_dl_mutex;
static std::string          g_dl_error;
static std::string          g_dl_temp_path;
static std::thread          g_dl_thread;

const model_manager::ModelInfo* model_manager::find_model(const std::string& id) {
    for (auto& m : g_known_models)
        if (m.id == id) return &m;
    return nullptr;
}

static std::string find_local_path(const model_manager::ModelInfo& info) {
    if (!g_local_dir.empty()) {
        fs::path p = fs::path(g_local_dir) / info.filename;
        if (fs::exists(p)) return p.string();
    }
    return {};
}

static std::string find_cached_path(const model_manager::ModelInfo& info) {
    if (!g_cache_dir.empty()) {
        fs::path p = fs::path(g_cache_dir) / info.filename;
        if (fs::exists(p)) return p.string();
    }
    return {};
}

void model_manager::init(const std::string& cache_dir, const std::string& local_dir) {
    g_cache_dir = cache_dir;
    g_local_dir = local_dir;
    if (!g_cache_dir.empty())
        fs::create_directories(g_cache_dir);
}

const std::vector<model_manager::ModelInfo>& model_manager::available_models() {
    return g_known_models;
}

bool model_manager::is_available(const std::string& model_id) {
    auto* info = find_model(model_id);
    if (!info) return false;
    return !find_local_path(*info).empty() || !find_cached_path(*info).empty();
}

std::string model_manager::model_path(const std::string& model_id) {
    auto* info = find_model(model_id);
    if (!info) return {};
    std::string local = find_local_path(*info);
    if (!local.empty()) return local;
    return find_cached_path(*info);
}

static void download_worker(std::string model_id) {
    auto* info = model_manager::find_model(model_id);
    if (!info) {
        std::lock_guard<std::mutex> lk(g_dl_mutex);
        g_dl_error = "Unknown model: " + model_id;
        g_dl_state.store(model_manager::DownloadState::Error);
        return;
    }

    fs::create_directories(g_cache_dir);
    std::string dest = (fs::path(g_cache_dir) / info->filename).string();
    std::string temp = dest + ".tmp";
    g_dl_temp_path = temp;

    std::string tag = DOWNLOAD_TAG;
    std::string url = "https://github.com/" + std::string(DOWNLOAD_REPO) + "/releases/"
        + (tag == "latest" ? "latest/download" : "download/" + tag)
        + "/" + info->filename;

    std::string cmd = "curl -fSL -o \"" + temp + "\" \"" + url + "\"";

    fprintf(stderr, "[reaper-stem-separation-plugin] downloading %s\n", url.c_str());

    std::atomic<bool> curl_done{false};
    std::thread monitor([&]() {
        while (!curl_done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (g_dl_cancel.load()) break;
            try {
                if (fs::exists(temp)) {
                    auto sz = fs::file_size(temp);
                    if (info->expected_bytes > 0)
                        g_dl_progress.store(static_cast<float>(sz) / static_cast<float>(info->expected_bytes));
                }
            } catch (...) {}
        }
    });

    int ret = std::system(cmd.c_str());
    curl_done.store(true);
    monitor.join();

    if (g_dl_cancel.load()) {
        fs::remove(temp);
        g_dl_state.store(model_manager::DownloadState::Idle);
        return;
    }

    if (ret != 0) {
        fs::remove(temp);
        std::lock_guard<std::mutex> lk(g_dl_mutex);
        if (ret == 127 || ret == 0x7F00)
            g_dl_error = "Download failed: curl not found. Please install curl and ensure it is in your PATH.";
        else
            g_dl_error = "Download failed (curl exit " + std::to_string(ret) + "). Check your internet connection.";
        g_dl_state.store(model_manager::DownloadState::Error);
        return;
    }

    std::error_code ec;
    fs::rename(temp, dest, ec);
    if (ec) {
        fs::remove(temp);
        std::lock_guard<std::mutex> lk(g_dl_mutex);
        g_dl_error = "Failed to save model: " + ec.message();
        g_dl_state.store(model_manager::DownloadState::Error);
        return;
    }

    g_dl_progress.store(1.0f);
    g_dl_state.store(model_manager::DownloadState::Done);
    fprintf(stderr, "[reaper-stem-separation-plugin] downloaded %s -> %s\n", info->filename.c_str(), dest.c_str());
}

void model_manager::start_download(const std::string& model_id) {
    if (g_dl_state.load() == DownloadState::Downloading) return;
    g_dl_state.store(DownloadState::Downloading);
    g_dl_cancel.store(false);
    g_dl_progress.store(0.0f);
    {
        std::lock_guard<std::mutex> lk(g_dl_mutex);
        g_dl_error.clear();
    }
    if (g_dl_thread.joinable()) g_dl_thread.join();
    g_dl_thread = std::thread(download_worker, model_id);
}

void model_manager::cancel_download() {
    g_dl_cancel.store(true);
}

model_manager::DownloadState model_manager::download_state() {
    return g_dl_state.load();
}

float model_manager::download_progress() {
    return g_dl_progress.load();
}

std::string model_manager::download_error() {
    std::lock_guard<std::mutex> lk(g_dl_mutex);
    return g_dl_error;
}

void model_manager::reset_download() {
    if (g_dl_thread.joinable()) g_dl_thread.join();
    g_dl_state.store(DownloadState::Idle);
    g_dl_progress.store(0.0f);
    g_dl_cancel.store(false);
    std::lock_guard<std::mutex> lk(g_dl_mutex);
    g_dl_error.clear();
}
