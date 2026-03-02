#ifndef SEPARATOR_H
#define SEPARATOR_H

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct SeparationRequest {
    std::string source_path;
    std::string model_path;
    double item_position = 0.0;
    double item_length = 0.0;
    int track_index = 0;
};

struct StemResult {
    std::string name;
    std::string path;
};

struct SeparationResult {
    std::vector<StemResult> stems;
    SeparationRequest request;
};

namespace separator {

enum class State { Idle, Running, Done, Error };

void start(const SeparationRequest& req);
void cancel();

State state();
float progress();
std::string status_message();
SeparationResult result();
std::string error_message();

void reset();
void cleanup_model();

} // namespace separator

#endif
