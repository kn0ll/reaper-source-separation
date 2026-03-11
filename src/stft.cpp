#include "stft.h"

#include <unsupported/Eigen/FFT>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

static constexpr float kWindowSumEpsilon = 1e-8f;

static Eigen::VectorXf hann_window(int size) {
    Eigen::VectorXf w(size);
    for (int i = 0; i < size; ++i)
        w(i) = 0.5f * (1.0f - std::cos(2.0f * std::numbers::pi_v<float> * i / size));
    return w;
}

Eigen::Tensor<float, 3> stft(const Eigen::VectorXf& signal, const STFTParams& params) {
    const int n_fft = params.n_fft;
    const int hop = params.hop_length;
    const int win_len = params.win_length;
    const int freq_bins = n_fft / 2 + 1;

    // Center-pad with reflection
    const int pad = n_fft / 2;
    const int padded_len = static_cast<int>(signal.size()) + 2 * pad;
    Eigen::VectorXf padded(padded_len);

    // Left reflection
    for (int i = 0; i < pad; ++i)
        padded(i) = signal(pad - i);
    // Center
    padded.segment(pad, signal.size()) = signal;
    // Right reflection
    for (int i = 0; i < pad; ++i) {
        int src = static_cast<int>(signal.size()) - 2 - i;
        padded(pad + static_cast<int>(signal.size()) + i) = (src >= 0) ? signal(src) : signal(0);
    }

    const int time_frames = (padded_len - n_fft) / hop + 1;
    auto window = hann_window(win_len);

    Eigen::FFT<float> fft;
    Eigen::Tensor<float, 3> result(freq_bins, time_frames, 2);

    std::vector<float> frame(n_fft);
    std::vector<std::complex<float>> spectrum;

    for (int t = 0; t < time_frames; ++t) {
        int start = t * hop;
        for (int i = 0; i < n_fft; ++i) {
            float sample = padded(start + i);
            float w = (i < win_len) ? window(i) : 0.0f;
            frame[i] = sample * w;
        }

        fft.fwd(spectrum, frame);

        for (int f = 0; f < freq_bins; ++f) {
            result(f, t, 0) = spectrum[f].real();
            result(f, t, 1) = spectrum[f].imag();
        }
    }

    return result;
}

Eigen::VectorXf istft(const Eigen::Tensor<float, 3>& spec, const STFTParams& params, int length) {
    const int n_fft = params.n_fft;
    const int hop = params.hop_length;
    const int win_len = params.win_length;
    const int freq_bins = spec.dimension(0);
    const int time_frames = spec.dimension(1);
    const int pad = n_fft / 2;

    auto window = hann_window(win_len);

    const int output_len = (time_frames - 1) * hop + n_fft;
    Eigen::VectorXf out = Eigen::VectorXf::Zero(output_len);
    Eigen::VectorXf window_sum = Eigen::VectorXf::Zero(output_len);

    Eigen::FFT<float> fft;
    std::vector<std::complex<float>> spectrum(n_fft);
    std::vector<float> frame;

    for (int t = 0; t < time_frames; ++t) {
        // Build full symmetric spectrum from onesided
        for (int f = 0; f < freq_bins; ++f)
            spectrum[f] = std::complex<float>(spec(f, t, 0), spec(f, t, 1));
        for (int f = freq_bins; f < n_fft; ++f) {
            int mirror = n_fft - f;
            spectrum[f] = std::conj(spectrum[mirror]);
        }

        fft.inv(frame, spectrum);

        int start = t * hop;
        for (int i = 0; i < n_fft; ++i) {
            float w = (i < win_len) ? window(i) : 0.0f;
            out(start + i) += frame[i] * w;
            window_sum(start + i) += w * w;
        }
    }

    // Normalize by squared window sum (Griffin-Lim COLA)
    for (int i = 0; i < output_len; ++i) {
        if (window_sum(i) > kWindowSumEpsilon)
            out(i) /= window_sum(i);
    }

    // Remove center padding
    int sig_len = output_len - 2 * pad;
    if (length > 0 && length <= sig_len)
        sig_len = length;

    Eigen::VectorXf result(sig_len);
    for (int i = 0; i < sig_len; ++i)
        result(i) = out(pad + i);

    return result;
}
