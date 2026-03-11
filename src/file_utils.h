#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open model: " + path);
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data(sz);
    if (!f.read(data.data(), sz))
        throw std::runtime_error("Failed to read model file");
    return data;
}

#endif
