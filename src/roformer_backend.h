#ifndef ROFORMER_BACKEND_H
#define ROFORMER_BACKEND_H

#include "backend.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <memory>
#include <string>

class RoFormerBackend : public ModelBackend {
public:
    struct Config {
        int chunk_size = 352800;    // samples per chunk (e.g. 8s at 44100)
        int num_overlap = 4;        // overlap factor (stride = chunk_size / num_overlap)
        int num_stems = 1;          // stems output by the ONNX model
        bool compute_residual = true; // add mix-minus residual as extra stem
    };

    void load(const std::string& model_path, bool use_gpu, const Config& config);
    Eigen::Tensor3dXf infer(const Eigen::MatrixXf& audio, ProgressCallback cb) override;

    bool is_loaded() const { return loaded_; }
    const std::string& loaded_path() const { return loaded_path_; }
    bool loaded_with_gpu() const { return loaded_gpu_; }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "roformer"};
    std::unique_ptr<Ort::Session> session_;
    Config config_;
    std::string loaded_path_;
    bool loaded_gpu_ = false;
    bool loaded_ = false;

    Eigen::Tensor3dXf run_inference(const Eigen::MatrixXf& audio, ProgressCallback cb);
};

#endif
