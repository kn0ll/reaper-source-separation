#ifndef ORT_MANAGER_H
#define ORT_MANAGER_H

#include <string>

namespace ort_manager {

void init(const std::string& ort_dir);

bool is_available();

void start_download();
void cancel_download();

enum class DownloadState { Idle, Downloading, Done, Error };
DownloadState download_state();
float download_progress();
std::string download_error();
void reset_download();

} // namespace ort_manager

#endif
