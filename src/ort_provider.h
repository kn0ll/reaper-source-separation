#ifndef ORT_PROVIDER_H
#define ORT_PROVIDER_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

bool try_cuda_provider(Ort::SessionOptions& opts);

#endif
