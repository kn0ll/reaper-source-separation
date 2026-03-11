#ifndef ORT_LOADER_H
#define ORT_LOADER_H

#include <string>

namespace ort_loader {

bool init(const std::string& ort_dir);
bool is_loaded();
const std::string& ort_dir();

} // namespace ort_loader

#endif
