#ifndef BACKEND_H
#define BACKEND_H

#include "tensor.hpp"
#include <Eigen/Dense>
#include <functional>
#include <string>

using ProgressCallback = std::function<void(float, const std::string&)>;

struct ModelBackend {
    virtual ~ModelBackend() = default;
    virtual Eigen::Tensor3dXf infer(const Eigen::MatrixXf& audio, ProgressCallback cb) = 0;

    bool is_loaded() const { return loaded_; }
    const std::string& loaded_path() const { return loaded_path_; }
    bool loaded_with_gpu() const { return loaded_gpu_; }

protected:
    std::string loaded_path_;
    bool loaded_gpu_ = false;
    bool loaded_ = false;
};

#endif
