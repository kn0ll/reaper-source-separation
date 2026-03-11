#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <string>
#include <vector>

namespace model_manager {

enum class BackendType { Demucs, RoFormer };

struct ModelInfo {
    std::string id;
    std::string filename;
    std::string display_name;
    std::string description;
    std::vector<std::string> stem_names;
    BackendType backend;
    int sample_rate;
    int chunk_size;
    int num_overlap;
    bool compute_residual;
    size_t expected_bytes;
};

void init(const std::string& cache_dir, const std::string& local_dir);

const std::vector<ModelInfo>& available_models();
const ModelInfo* find_model(const std::string& model_id);

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
