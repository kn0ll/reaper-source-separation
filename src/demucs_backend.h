#ifndef DEMUCS_BACKEND_H
#define DEMUCS_BACKEND_H

#include "backend.h"
#include "demucs.hpp"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <memory>
#include <string>

class DemucsBackend : public ModelBackend {
public:
    void load(const std::string& model_path, bool use_gpu);
    Eigen::Tensor3dXf infer(const Eigen::MatrixXf& audio, ProgressCallback cb) override;

    bool is_loaded() const { return loaded_; }
    const std::string& loaded_path() const { return loaded_path_; }
    bool loaded_with_gpu() const { return loaded_gpu_; }
    int num_sources() const { return model_.nb_sources; }

private:
    demucsonnx::demucs_model model_;
    std::string loaded_path_;
    bool loaded_gpu_ = false;
    bool loaded_ = false;
};

#endif
