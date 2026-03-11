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
};

#endif
