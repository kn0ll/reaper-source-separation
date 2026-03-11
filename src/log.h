#ifndef LOG_H
#define LOG_H

#include <cstdio>

#define LOG(fmt, ...) fprintf(stderr, "[reaper-stem-separation-plugin] " fmt, ##__VA_ARGS__)

#endif
