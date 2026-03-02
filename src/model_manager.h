#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <string>
#include <vector>

namespace model_manager {

struct ModelInfo {
    std::string id;
    std::string filename;
    std::string display_name;
    std::string stems;
    size_t expected_bytes;
};

void init(const std::string& cache_dir, const std::string& local_dir);

const std::vector<ModelInfo>& available_models();

bool is_available(const std::string& model_id);

std::string model_path(const std::string& model_id);

void start_download(const std::string& model_id);
void cancel_download();

enum class DownloadState { Idle, Downloading, Done, Error };
DownloadState download_state();
float download_progress();
std::string download_error();
void reset_download();

} // namespace model_manager

#endif
