#pragma once
// ============================================================
// common/ppm.h - Minimal PPM/PGM image I/O
// ============================================================
// Pure C++ (no CUDA, no libpng, no OpenCV).
//
// The image exercises need to read and write pictures, but adding
// an image library would make the repository harder to build. The
// Netpbm formats solve this: a 15-line header parser is all it
// takes, and every image viewer plus ImageMagick can open them.
//
//   P6 = binary RGB   (3 bytes per pixel)
//   P5 = binary gray  (1 byte per pixel)
//
// Convert to PNG afterwards if you like:
//   convert out.ppm out.png        # ImageMagick
// ============================================================

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;                   // 3 = RGB, 1 = grayscale
    std::vector<unsigned char> data;    // row-major, interleaved channels

    size_t pixelCount() const { return static_cast<size_t>(width) * height; }
    size_t byteCount() const { return pixelCount() * static_cast<size_t>(channels); }

    void resize(int w, int h, int c) {
        width = w;
        height = h;
        channels = c;
        data.assign(byteCount(), 0);
    }
};

// ------------------------------------------------------------
// Header parsing
// ------------------------------------------------------------
// Netpbm headers allow whitespace and '#' comments anywhere
// between tokens, so tokens must be read one at a time.
inline bool ppmReadToken(FILE* f, std::string& out) {
    out.clear();
    int c = fgetc(f);
    for (;;) {
        while (c != EOF && std::isspace(c)) c = fgetc(f);
        if (c == '#') {  // comment runs to end of line
            while (c != EOF && c != '\n') c = fgetc(f);
            continue;
        }
        break;
    }
    if (c == EOF) return false;
    while (c != EOF && !std::isspace(c)) {
        out.push_back(static_cast<char>(c));
        c = fgetc(f);
    }
    return !out.empty();
}

inline bool loadPPM(const char* path, Image& img) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "loadPPM: cannot open '%s'\n", path);
        return false;
    }

    std::string magic, w, h, maxv;
    if (!ppmReadToken(f, magic) || !ppmReadToken(f, w) || !ppmReadToken(f, h) ||
        !ppmReadToken(f, maxv)) {
        fprintf(stderr, "loadPPM: malformed header in '%s'\n", path);
        fclose(f);
        return false;
    }

    int channels = 0;
    if (magic == "P6") channels = 3;
    else if (magic == "P5") channels = 1;
    else {
        fprintf(stderr, "loadPPM: '%s' is '%s', only binary P5/P6 are supported\n", path,
                magic.c_str());
        fclose(f);
        return false;
    }
    if (std::atoi(maxv.c_str()) != 255) {
        fprintf(stderr, "loadPPM: only 8-bit images (maxval 255) are supported\n");
        fclose(f);
        return false;
    }

    img.resize(std::atoi(w.c_str()), std::atoi(h.c_str()), channels);
    size_t got = fread(img.data.data(), 1, img.byteCount(), f);
    fclose(f);
    if (got != img.byteCount()) {
        fprintf(stderr, "loadPPM: truncated pixel data in '%s' (%zu of %zu bytes)\n", path,
                got, img.byteCount());
        return false;
    }
    return true;
}

inline bool savePPM(const char* path, const Image& img) {
    if (img.channels != 1 && img.channels != 3) {
        fprintf(stderr, "savePPM: unsupported channel count %d\n", img.channels);
        return false;
    }
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "savePPM: cannot write '%s'\n", path);
        return false;
    }
    fprintf(f, "%s\n%d %d\n255\n", img.channels == 3 ? "P6" : "P5", img.width, img.height);
    fwrite(img.data.data(), 1, img.byteCount(), f);
    fclose(f);
    return true;
}

// ------------------------------------------------------------
// Synthetic test image
// ------------------------------------------------------------
// So the exercises run out of the box without shipping binary
// assets. The pattern deliberately mixes smooth gradients (which
// blur invisibly) with hard edges and a checkerboard (which make
// filter bugs obvious).
inline Image makeTestImage(int width = 1024, int height = 768) {
    Image img;
    img.resize(width, height, 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t i = (static_cast<size_t>(y) * width + x) * 3;
            unsigned char r = static_cast<unsigned char>(255 * x / (width - 1));
            unsigned char g = static_cast<unsigned char>(255 * y / (height - 1));
            unsigned char b = 64;

            // Checkerboard of 32x32 tiles.
            if (((x / 32) + (y / 32)) % 2 == 0) b = 200;

            // A filled circle with a hard edge in the centre.
            double dx = x - width * 0.5, dy = y - height * 0.5;
            double radius = (height < width ? height : width) * 0.25;
            if (dx * dx + dy * dy < radius * radius) {
                r = 250;
                g = 250;
                b = 250;
            }

            img.data[i + 0] = r;
            img.data[i + 1] = g;
            img.data[i + 2] = b;
        }
    }
    return img;
}

// Load `path` if it exists, otherwise generate the synthetic image
// and write it out so the user can look at the input too.
inline Image loadOrCreateTestImage(const char* path, int width = 1024, int height = 768) {
    Image img;
    if (loadPPM(path, img)) {
        printf("  Loaded image: %s (%dx%d, %d channels)\n", path, img.width, img.height,
               img.channels);
        return img;
    }
    printf("  Generating synthetic test image %dx%d -> %s\n", width, height, path);
    img = makeTestImage(width, height);
    savePPM(path, img);
    return img;
}
