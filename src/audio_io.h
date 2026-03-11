#ifndef AUDIO_IO_H
#define AUDIO_IO_H

#include "separator.h"
#include "tensor.hpp"
#include <string>
#include <vector>

namespace audio_io {

Eigen::MatrixXf load(const std::string& path, int expected_sample_rate);

std::vector<StemResult> write_stems(
    const Eigen::Tensor3dXf& targets,
    const std::vector<std::string>& stem_names,
    int sample_rate,
    const std::string& output_dir);

} // namespace audio_io

#endif
