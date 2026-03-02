#include "audio_io.h"
#include "demucs.hpp"
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <Eigen/Dense>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace nqr;

static const char* stem_name(int idx) {
    static const char* names[] = {
        "drums", "bass", "other", "vocals", "guitar", "piano"
    };
    if (idx >= 0 && idx < 6) return names[idx];
    return "unknown";
}

Eigen::MatrixXf audio_io::load(const std::string& path) {
    auto file_data = std::make_shared<AudioData>();
    NyquistIO loader;
    loader.Load(file_data.get(), path);

    if (file_data->sampleRate != demucsonnx::SUPPORTED_SAMPLE_RATE) {
        throw std::runtime_error(
            "Unsupported sample rate: " + std::to_string(file_data->sampleRate)
            + " (need " + std::to_string(demucsonnx::SUPPORTED_SAMPLE_RATE) + ")");
    }

    if (file_data->channelCount != 2 && file_data->channelCount != 1) {
        throw std::runtime_error("Only mono and stereo audio supported");
    }

    std::size_t N = file_data->samples.size() / file_data->channelCount;
    Eigen::MatrixXf audio(2, N);

    if (file_data->channelCount == 1) {
        for (std::size_t i = 0; i < N; ++i) {
            audio(0, i) = file_data->samples[i];
            audio(1, i) = file_data->samples[i];
        }
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            audio(0, i) = file_data->samples[2 * i];
            audio(1, i) = file_data->samples[2 * i + 1];
        }
    }

    return audio;
}

std::vector<StemResult> audio_io::write_stems(
    const Eigen::Tensor3dXf& targets,
    int nb_sources,
    const std::string& output_dir)
{
    fs::create_directories(output_dir);
    std::vector<StemResult> results;
    long num_samples = targets.dimension(2);

    for (int t = 0; t < nb_sources; ++t) {
        std::string name = stem_name(t);
        fs::path out_path = fs::path(output_dir) / (name + ".wav");

        auto file_data = std::make_shared<AudioData>();
        file_data->sampleRate = demucsonnx::SUPPORTED_SAMPLE_RATE;
        file_data->channelCount = 2;
        file_data->samples.resize(num_samples * 2);

        for (long s = 0; s < num_samples; ++s) {
            file_data->samples[2 * s]     = targets(t, 0, s);
            file_data->samples[2 * s + 1] = targets(t, 1, s);
        }

        encode_wav_to_disk(
            {file_data->channelCount, PCM_FLT, DITHER_TRIANGLE},
            file_data.get(), out_path.string());

        results.push_back({name, out_path.string()});
    }

    return results;
}
