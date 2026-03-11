#include "ort_provider.h"
#include "log.h"
#include <algorithm>
#include <string>
#include <vector>

bool try_cuda_provider(Ort::SessionOptions& opts) {
    auto providers = Ort::GetAvailableProviders();
    std::string provider_list;
    for (const auto& p : providers) {
        if (!provider_list.empty()) provider_list += ' ';
        provider_list += p;
    }
    LOG("available_providers=[%s]\n", provider_list.c_str());

    if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end())
        return false;
    try {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        LOG("CUDA provider attached\n");
        return true;
    } catch (const std::exception& e) {
        LOG("CUDA provider failed: %s\n", e.what());
        return false;
    } catch (...) {
        LOG("CUDA provider failed (unknown error)\n");
        return false;
    }
}
