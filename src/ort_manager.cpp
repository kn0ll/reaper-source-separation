#include "ort_manager.h"
#include "ort_loader.h"
#include "log.h"
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;

#ifndef ORT_VERSION
#define ORT_VERSION "1.19.2"
#endif

static std::string g_ort_dir;

static std::atomic<ort_manager::DownloadState> g_dl_state{ort_manager::DownloadState::Idle};
static std::atomic<float>   g_dl_progress{0.0f};
static std::atomic<bool>    g_dl_cancel{false};
static std::mutex           g_dl_mutex;
static std::string          g_dl_error;
static std::thread          g_dl_thread;

struct OrtAssetInfo {
    const char* asset_name;
    bool is_tgz;
    size_t expected_bytes;
};

static OrtAssetInfo platform_asset() {
#if defined(__APPLE__)
    #if defined(__aarch64__) || defined(__arm64__)
    return {"onnxruntime-osx-arm64-" ORT_VERSION ".tgz", true, 30'000'000};
    #else
    return {"onnxruntime-osx-x86_64-" ORT_VERSION ".tgz", true, 30'000'000};
    #endif
#elif defined(_WIN32)
    return {"onnxruntime-win-x64-" ORT_VERSION ".zip", false, 30'000'000};
#else
    return {"onnxruntime-linux-x64-" ORT_VERSION ".tgz", true, 30'000'000};
#endif
}

static void download_worker() {
    auto asset = platform_asset();
    std::string url = "https://github.com/microsoft/onnxruntime/releases/download/v"
        + std::string(ORT_VERSION) + "/" + asset.asset_name;

    fs::create_directories(g_ort_dir);

    std::string temp_archive = (fs::path(g_ort_dir) / "ort_download.tmp").string();

    std::string dl_cmd = "curl -fSL -o \"" + temp_archive + "\" \"" + url + "\"";

    LOG("ort_manager: downloading url=%s\n", url.c_str());

    std::atomic<bool> curl_done{false};
    std::thread monitor([&]() {
        while (!curl_done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (g_dl_cancel.load()) break;
            try {
                if (fs::exists(temp_archive)) {
                    auto sz = fs::file_size(temp_archive);
                    if (asset.expected_bytes > 0)
                        g_dl_progress.store(
                            static_cast<float>(sz) / static_cast<float>(asset.expected_bytes) * 0.9f);
                }
            } catch (...) {}
        }
    });

    int ret = std::system(dl_cmd.c_str());
    curl_done.store(true);
    monitor.join();

    if (g_dl_cancel.load()) {
        fs::remove(temp_archive);
        g_dl_state.store(ort_manager::DownloadState::Idle);
        return;
    }

    if (ret != 0) {
        fs::remove(temp_archive);
        std::lock_guard<std::mutex> lk(g_dl_mutex);
        if (ret == 127 || ret == 0x7F00)
            g_dl_error = "Download failed: curl not found. Please install curl and ensure it is in your PATH.";
        else
            g_dl_error = "Download failed (curl exit " + std::to_string(ret) + "). Check your internet connection.";
        LOG("ort_manager: download failed curl_exit=%d\n", ret);
        g_dl_state.store(ort_manager::DownloadState::Error);
        return;
    }

    g_dl_progress.store(0.9f);

    // Extract the archive. ORT archives have a top-level directory
    // (e.g. onnxruntime-linux-x64-1.19.2/) with lib/ inside it.
    // Extract to a temp dir, then copy just the library files out.
    std::string tmpdir = (fs::path(g_ort_dir) / "_ort_extract").string();
    fs::create_directories(tmpdir);

    std::string extract_cmd;
#ifdef _WIN32
    extract_cmd = "tar -xf \"" + temp_archive + "\" -C \"" + tmpdir + "\"";
#else
    if (asset.is_tgz) {
        extract_cmd = "tar xzf \"" + temp_archive + "\" -C \"" + tmpdir + "\"";
    } else {
        extract_cmd = "unzip -q -o \"" + temp_archive + "\" -d \"" + tmpdir + "\"";
    }
#endif

    LOG("ort_manager: extracting cmd=%s\n", extract_cmd.c_str());
    ret = std::system(extract_cmd.c_str());

    fs::remove(temp_archive);

    if (ret != 0) {
        std::error_code ec;
        fs::remove_all(tmpdir, ec);
        std::lock_guard<std::mutex> lk(g_dl_mutex);
        g_dl_error = "Failed to extract ONNX Runtime (exit " + std::to_string(ret) + ").";
        LOG("ort_manager: extraction failed exit=%d\n", ret);
        g_dl_state.store(ort_manager::DownloadState::Error);
        return;
    }

    // Copy library files from the extracted archive's lib/ directory
    try {
        for (auto& entry : fs::recursive_directory_iterator(tmpdir)) {
            if (!entry.is_regular_file() && !entry.is_symlink()) continue;
            if (entry.path().parent_path().filename() != "lib") continue;
            std::string fname = entry.path().filename().string();
            if (fname.find("onnxruntime") == std::string::npos) continue;
            fs::copy(entry.path(), fs::path(g_ort_dir) / fname,
                     fs::copy_options::overwrite_existing);
        }
    } catch (const std::exception& e) {
        std::error_code ec;
        fs::remove_all(tmpdir, ec);
        std::lock_guard<std::mutex> lk(g_dl_mutex);
        g_dl_error = std::string("Failed to copy ORT libraries: ") + e.what();
        LOG("ort_manager: copy failed: %s\n", e.what());
        g_dl_state.store(ort_manager::DownloadState::Error);
        return;
    }

    {
        std::error_code ec;
        fs::remove_all(tmpdir, ec);
    }

    if (!ort_loader::init(g_ort_dir)) {
        std::lock_guard<std::mutex> lk(g_dl_mutex);
        g_dl_error = "ONNX Runtime was downloaded but failed to load.";
        LOG("ort_manager: ort_loader::init failed after extraction\n");
        g_dl_state.store(ort_manager::DownloadState::Error);
        return;
    }

    g_dl_progress.store(1.0f);
    g_dl_state.store(ort_manager::DownloadState::Done);
    LOG("ort_manager: download and extraction complete dir=%s\n", g_ort_dir.c_str());
}

void ort_manager::init(const std::string& ort_dir) {
    g_ort_dir = ort_dir;
    fs::create_directories(g_ort_dir);
    ort_loader::init(g_ort_dir);
    LOG("ort_manager: init dir=%s available=%s\n",
        g_ort_dir.c_str(), ort_loader::is_loaded() ? "yes" : "no");
}

bool ort_manager::is_available() {
    return ort_loader::is_loaded();
}

void ort_manager::start_download() {
    if (g_dl_state.load() == DownloadState::Downloading) return;
    g_dl_state.store(DownloadState::Downloading);
    g_dl_cancel.store(false);
    g_dl_progress.store(0.0f);
    {
        std::lock_guard<std::mutex> lk(g_dl_mutex);
        g_dl_error.clear();
    }
    if (g_dl_thread.joinable()) g_dl_thread.join();
    g_dl_thread = std::thread(download_worker);
}

void ort_manager::cancel_download() {
    g_dl_cancel.store(true);
}

ort_manager::DownloadState ort_manager::download_state() {
    return g_dl_state.load();
}

float ort_manager::download_progress() {
    return g_dl_progress.load();
}

std::string ort_manager::download_error() {
    std::lock_guard<std::mutex> lk(g_dl_mutex);
    return g_dl_error;
}

void ort_manager::reset_download() {
    if (g_dl_thread.joinable()) g_dl_thread.join();
    g_dl_state.store(DownloadState::Idle);
    g_dl_progress.store(0.0f);
    g_dl_cancel.store(false);
    std::lock_guard<std::mutex> lk(g_dl_mutex);
    g_dl_error.clear();
}
