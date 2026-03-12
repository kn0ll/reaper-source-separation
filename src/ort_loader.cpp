#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include "ort_loader.h"
#include "log.h"
#include <new>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static const OrtApiBase* g_api_base = nullptr;
static bool g_loaded = false;
static std::string g_ort_dir;

// Stub ORT API installed before vendor static init so that vendor globals
// (Ort::AllocatorWithDefaultOptions, Ort::RunOptions) can construct safely
// with null internal pointers instead of crashing.
static OrtStatus* ORT_API_CALL StubGetAllocatorWithDefaultOptions(OrtAllocator** out) NO_EXCEPTION {
    *out = nullptr;
    return nullptr;
}

static OrtStatus* ORT_API_CALL StubCreateRunOptions(OrtRunOptions** out) NO_EXCEPTION {
    *out = nullptr;
    return nullptr;
}

static OrtApi g_stub_api = {};

struct OrtStubInit {
    OrtStubInit() {
        g_stub_api.GetAllocatorWithDefaultOptions = StubGetAllocatorWithDefaultOptions;
        g_stub_api.CreateRunOptions = StubCreateRunOptions;
        Ort::InitApi(&g_stub_api);
    }
};

#ifdef _MSC_VER
#pragma warning(suppress : 4073)
#pragma init_seg(lib)
static OrtStubInit g_ort_stub_init;
#else
__attribute__((init_priority(101)))
static OrtStubInit g_ort_stub_init;
#endif

ORT_EXPORT const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION {
    return g_api_base;
}

// Vendor globals declared in vendor/demucs.onnx/src/demucs.hpp.
// They were constructed during static init with the stub API (null internals).
// After loading the real ORT, reconstruct them with placement new.
namespace demucsonnx {
    extern Ort::AllocatorWithDefaultOptions allocator;
    extern Ort::RunOptions run_options;
}

static void reinit_vendor_globals() {
    new (&demucsonnx::allocator) Ort::AllocatorWithDefaultOptions();
    new (&demucsonnx::run_options) Ort::RunOptions();
}

bool ort_loader::init(const std::string& ort_dir) {
    if (g_loaded) return true;
    g_ort_dir = ort_dir;

#ifdef _WIN32
    std::string lib_path = ort_dir + "\\onnxruntime.dll";
    SetDllDirectoryA(ort_dir.c_str());
    HMODULE handle = LoadLibraryA(lib_path.c_str());
    SetDllDirectoryA(nullptr);
    if (!handle) {
        LOG("ort_loader: LoadLibrary failed for %s (error %lu)\n",
            lib_path.c_str(), GetLastError());
        return false;
    }
    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)(void);
    auto fn = reinterpret_cast<GetApiBaseFn>(GetProcAddress(handle, "OrtGetApiBase"));
#else
#ifdef __APPLE__
    std::string lib_path = ort_dir + "/libonnxruntime.dylib";
#else
    std::string lib_path = ort_dir + "/libonnxruntime.so";
#endif
    void* handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        LOG("ort_loader: dlopen failed: %s\n", dlerror());
        return false;
    }
    using GetApiBaseFn = const OrtApiBase*(*)(void);
    auto fn = reinterpret_cast<GetApiBaseFn>(dlsym(handle, "OrtGetApiBase"));
#endif

    if (!fn) {
        LOG("ort_loader: could not resolve OrtGetApiBase\n");
        return false;
    }

    g_api_base = fn();
    if (!g_api_base) {
        LOG("ort_loader: OrtGetApiBase returned null\n");
        return false;
    }

    const OrtApi* api = g_api_base->GetApi(ORT_API_VERSION);
    if (!api) {
        LOG("ort_loader: GetApi(%d) returned null\n", ORT_API_VERSION);
        return false;
    }

    Ort::InitApi(api);
    reinit_vendor_globals();
    g_loaded = true;
    LOG("ort_loader: loaded from %s\n", lib_path.c_str());
    return true;
}

bool ort_loader::is_loaded() {
    return g_loaded;
}

const std::string& ort_loader::ort_dir() {
    return g_ort_dir;
}
