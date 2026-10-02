#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static uint32_t read_be32(FILE* f) {
    unsigned char b[4];
    if (std::fread(b, 1, 4, f) != 4) { std::fprintf(stderr, "read error\n"); std::exit(1); }
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
}

// Returns count * 784 floats in [0,1], one image after another
static std::vector<float> load_images(const char* path, int& count) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    read_be32(f);                         // magic number (ignored)
    count = (int)read_be32(f);
    uint32_t rows = read_be32(f), cols = read_be32(f);
    std::vector<unsigned char> raw((size_t)count * rows * cols);
    if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) { std::fprintf(stderr, "short read\n"); std::exit(1); }
    std::fclose(f);
    std::vector<float> out(raw.size());
    for (size_t i = 0; i < raw.size(); i++) out[i] = raw[i] / 255.0f;
    return out;
}

// Returns count labels, each 0..9
static std::vector<int> load_labels(const char* path, int& count) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    read_be32(f);                         // magic number (ignored)
    count = (int)read_be32(f);
    std::vector<unsigned char> raw(count);
    if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) { std::fprintf(stderr, "short read\n"); std::exit(1); }
    std::fclose(f);
    return std::vector<int>(raw.begin(), raw.end());
}