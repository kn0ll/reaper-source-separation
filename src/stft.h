#ifndef STFT_H
#define STFT_H

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

struct STFTParams {
    int n_fft = 2048;
    int hop_length = 512;
    int win_length = 2048;
};

// Forward STFT: real signal -> complex spectrogram stored as real tensor
// Input:  signal of length N (single channel)
// Output: (freq_bins, time_frames, 2) where last dim is [real, imag]
//         freq_bins = n_fft/2 + 1
Eigen::Tensor<float, 3> stft(const Eigen::VectorXf& signal, const STFTParams& params);

// Inverse STFT: complex spectrogram -> real signal
// Input:  (freq_bins, time_frames, 2) where last dim is [real, imag]
// Output: signal of length `length` (or auto-computed if length <= 0)
Eigen::VectorXf istft(const Eigen::Tensor<float, 3>& spec, const STFTParams& params, int length = -1);

#endif
