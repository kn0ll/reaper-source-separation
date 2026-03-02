#ifndef AUDIO_IO_H
#define AUDIO_IO_H

#include "separator.h"
#include "tensor.hpp"
#include <string>
#include <vector>

namespace audio_io {

Eigen::MatrixXf load(const std::string& path);

std::vector<StemResult> write_stems(
    const Eigen::Tensor3dXf& targets,
    int nb_sources,
    const std::string& output_dir);

} // namespace audio_io

#endif
