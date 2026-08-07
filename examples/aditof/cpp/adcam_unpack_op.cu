#include "adcam_unpack_op.hpp"

#include <algorithm>

//==============================================================================
//  JET COLOR MAP (256 × 3) stored in constant memory
//------------------------------------------------------------------------------
//  • Constant memory is cached and optimized for broadcast reads.
//  • Each entry is RGB for a normalized value 0–255.
//  • Used by jet_kernel() to convert depth → pseudo‑color.
//==============================================================================
__constant__ uint8_t JET_LUT[256 * 3] = {
    0x00,0x00,0x7F, 0x00,0x00,0x84, 0x00,0x00,0x88, 0x00,0x00,0x8D,
    0x00,0x00,0x91, 0x00,0x00,0x96, 0x00,0x00,0x9A, 0x00,0x00,0x9F,
    0x00,0x00,0xA3, 0x00,0x00,0xA8, 0x00,0x00,0xAC, 0x00,0x00,0xB1,
    0x00,0x00,0xB6, 0x00,0x00,0xBA, 0x00,0x00,0xBF, 0x00,0x00,0xC3,
    0x00,0x00,0xC8, 0x00,0x00,0xCC, 0x00,0x00,0xD1, 0x00,0x00,0xD5,
    0x00,0x00,0xDA, 0x00,0x00,0xDE, 0x00,0x00,0xE3, 0x00,0x00,0xE8,
    0x00,0x00,0xEC, 0x00,0x00,0xF1, 0x00,0x00,0xF5, 0x00,0x00,0xFA,
    0x00,0x00,0xFE, 0x00,0x00,0xFF, 0x00,0x00,0xFF, 0x00,0x00,0xFF,
    0x00,0x00,0xFF, 0x00,0x04,0xFF, 0x00,0x08,0xFF, 0x00,0x0C,0xFF,
    0x00,0x10,0xFF, 0x00,0x14,0xFF, 0x00,0x18,0xFF, 0x00,0x1C,0xFF,
    0x00,0x20,0xFF, 0x00,0x24,0xFF, 0x00,0x28,0xFF, 0x00,0x2C,0xFF,
    0x00,0x30,0xFF, 0x00,0x34,0xFF, 0x00,0x38,0xFF, 0x00,0x3C,0xFF,
    0x00,0x40,0xFF, 0x00,0x44,0xFF, 0x00,0x48,0xFF, 0x00,0x4C,0xFF,
    0x00,0x50,0xFF, 0x00,0x54,0xFF, 0x00,0x58,0xFF, 0x00,0x5C,0xFF,
    0x00,0x60,0xFF, 0x00,0x64,0xFF, 0x00,0x68,0xFF, 0x00,0x6C,0xFF,
    0x00,0x70,0xFF, 0x00,0x74,0xFF, 0x00,0x78,0xFF, 0x00,0x7C,0xFF,
    0x00,0x80,0xFF, 0x00,0x84,0xFF, 0x00,0x88,0xFF, 0x00,0x8C,0xFF,
    0x00,0x90,0xFF, 0x00,0x94,0xFF, 0x00,0x98,0xFF, 0x00,0x9C,0xFF,
    0x00,0xA0,0xFF, 0x00,0xA4,0xFF, 0x00,0xA8,0xFF, 0x00,0xAC,0xFF,
    0x00,0xB0,0xFF, 0x00,0xB4,0xFF, 0x00,0xB8,0xFF, 0x00,0xBC,0xFF,
    0x00,0xC0,0xFF, 0x00,0xC4,0xFF, 0x00,0xC8,0xFF, 0x00,0xCC,0xFF,
    0x00,0xD0,0xFF, 0x00,0xD4,0xFF, 0x00,0xD8,0xFF, 0x00,0xDC,0xFE,
    0x00,0xE0,0xFA, 0x00,0xE4,0xF7, 0x02,0xE8,0xF4, 0x05,0xEC,0xF1,
    0x08,0xF0,0xED, 0x0C,0xF4,0xEA, 0x0F,0xF8,0xE7, 0x12,0xFC,0xE4,
    0x15,0xFF,0xE1, 0x18,0xFF,0xDD, 0x1C,0xFF,0xDA, 0x1F,0xFF,0xD7,
    0x22,0xFF,0xD4, 0x25,0xFF,0xD0, 0x29,0xFF,0xCD, 0x2C,0xFF,0xCA,
    0x2F,0xFF,0xC7, 0x32,0xFF,0xC3, 0x36,0xFF,0xC0, 0x39,0xFF,0xBD,
    0x3C,0xFF,0xBA, 0x3F,0xFF,0xB7, 0x42,0xFF,0xB3, 0x46,0xFF,0xB0,
    0x49,0xFF,0xAD, 0x4C,0xFF,0xAA, 0x4F,0xFF,0xA6, 0x53,0xFF,0xA3,
    0x56,0xFF,0xA0, 0x59,0xFF,0x9D, 0x5C,0xFF,0x9A, 0x5F,0xFF,0x96,
    0x63,0xFF,0x93, 0x66,0xFF,0x90, 0x69,0xFF,0x8D, 0x6C,0xFF,0x89,
    0x70,0xFF,0x86, 0x73,0xFF,0x83, 0x76,0xFF,0x80, 0x79,0xFF,0x7D,
    0x7C,0xFF,0x79, 0x80,0xFF,0x76, 0x83,0xFF,0x73, 0x86,0xFF,0x70,
    0x89,0xFF,0x6C, 0x8D,0xFF,0x69, 0x90,0xFF,0x66, 0x93,0xFF,0x63,
    0x96,0xFF,0x5F, 0x9A,0xFF,0x5C, 0x9D,0xFF,0x59, 0xA0,0xFF,0x56,
    0xA3,0xFF,0x53, 0xA6,0xFF,0x4F, 0xAA,0xFF,0x4C, 0xAD,0xFF,0x49,
    0xB0,0xFF,0x46, 0xB3,0xFF,0x42, 0xB7,0xFF,0x3F, 0xBA,0xFF,0x3C,
    0xBD,0xFF,0x39, 0xC0,0xFF,0x36, 0xC3,0xFF,0x32, 0xC7,0xFF,0x2F,
    0xCA,0xFF,0x2C, 0xCD,0xFF,0x29, 0xD0,0xFF,0x25, 0xD4,0xFF,0x22,
    0xD7,0xFF,0x1F, 0xDA,0xFF,0x1C, 0xDD,0xFF,0x18, 0xE0,0xFF,0x15,
    0xE4,0xFF,0x12, 0xE7,0xFF,0x0F, 0xEA,0xFF,0x0C, 0xED,0xFF,0x08,
    0xF1,0xFC,0x05, 0xF4,0xF8,0x02, 0xF7,0xF4,0x00, 0xFA,0xF0,0x00,
    0xFE,0xED,0x00, 0xFF,0xE9,0x00, 0xFF,0xE5,0x00, 0xFF,0xE2,0x00,
    0xFF,0xDE,0x00, 0xFF,0xDA,0x00, 0xFF,0xD7,0x00, 0xFF,0xD3,0x00,
    0xFF,0xCF,0x00, 0xFF,0xCB,0x00, 0xFF,0xC8,0x00, 0xFF,0xC4,0x00,
    0xFF,0xC0,0x00, 0xFF,0xBD,0x00, 0xFF,0xB9,0x00, 0xFF,0xB5,0x00,
    0xFF,0xB1,0x00, 0xFF,0xAE,0x00, 0xFF,0xAA,0x00, 0xFF,0xA6,0x00,
    0xFF,0xA3,0x00, 0xFF,0x9F,0x00, 0xFF,0x9B,0x00, 0xFF,0x98,0x00,
    0xFF,0x94,0x00, 0xFF,0x90,0x00, 0xFF,0x8C,0x00, 0xFF,0x89,0x00,
    0xFF,0x85,0x00, 0xFF,0x81,0x00, 0xFF,0x7E,0x00, 0xFF,0x7A,0x00,
    0xFF,0x76,0x00, 0xFF,0x73,0x00, 0xFF,0x6F,0x00, 0xFF,0x6B,0x00,
    0xFF,0x67,0x00, 0xFF,0x64,0x00, 0xFF,0x60,0x00, 0xFF,0x5C,0x00,
    0xFF,0x59,0x00, 0xFF,0x55,0x00, 0xFF,0x51,0x00, 0xFF,0x4D,0x00,
    0xFF,0x4A,0x00, 0xFF,0x46,0x00, 0xFF,0x42,0x00, 0xFF,0x3F,0x00,
    0xFF,0x3B,0x00, 0xFF,0x37,0x00, 0xFF,0x34,0x00, 0xFF,0x30,0x00,
    0xFF,0x2C,0x00, 0xFF,0x28,0x00, 0xFF,0x25,0x00, 0xFF,0x21,0x00,
    0xFF,0x1D,0x00, 0xFF,0x1A,0x00, 0xFF,0x16,0x00, 0xFE,0x12,0x00,
    0xFA,0x0F,0x00, 0xF5,0x0B,0x00, 0xF1,0x07,0x00, 0xEC,0x03,0x00,
    0xE8,0x00,0x00, 0xE3,0x00,0x00, 0xDE,0x00,0x00, 0xDA,0x00,0x00,
    0xD5,0x00,0x00, 0xD1,0x00,0x00, 0xCC,0x00,0x00, 0xC8,0x00,0x00,
    0xC3,0x00,0x00, 0xBF,0x00,0x00, 0xBA,0x00,0x00, 0xB6,0x00,0x00,
    0xB1,0x00,0x00, 0xAC,0x00,0x00, 0xA8,0x00,0x00, 0xA3,0x00,0x00,
    0x9F,0x00,0x00, 0x9A,0x00,0x00, 0x96,0x00,0x00, 0x91,0x00,0x00,
    0x8D,0x00,0x00, 0x88,0x00,0x00, 0x84,0x00,0x00, 0x7F,0x00,0x00
};


//==============================================================================
//  KERNEL: shift_and_cast_kernel
//------------------------------------------------------------------------------
//  Converts 16‑bit values → 8‑bit by shifting right 8 bits.
//  Equivalent to Python: (cp_frame >> 8).astype(cp.uint8)
//------------------------------------------------------------------------------
//  in   : uint16_t*  (device)  — input 16‑bit values
//  out  : uint8_t*   (device)  — output 8‑bit values
//  count: number of elements
//==============================================================================
__global__
void shift_and_cast_kernel(const uint16_t* in,
                           uint8_t* out,
                           int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = static_cast<uint8_t>(in[idx] >> 8);
    }
}



//==============================================================================
//  KERNEL: unpack_kernel
//------------------------------------------------------------------------------
//  Unpacks ADI ToF 5-bytes/pixel packed format into three separate planes.
//  Handles both MP and QMP modes.
//
//  Frame structure — same for MP and QMP:
//
//  Subframe 1 — Depth + Confidence interleaved (3 bytes/pixel):
//      Byte 0: depth LSB
//      Byte 1: depth MSB  → depth[i] = uint16 little-endian
//      Byte 2: conf       → conf[i]  = uint8
//
//  Subframe 2 — Active Brightness only (2 bytes/pixel):
//      Byte 0: ab LSB
//      Byte 1: ab MSB     → ab[i] = uint16 little-endian
//
//  Memory layout (N = pixel_width × pixel_height pixels; total = N × 5 bytes):
//
//   ┌─────────────────────────────────────────────────────┐
//   │  Subframe 1: [D1_L][D1_H][C1] [D2_L][D2_H][C2]...   │  3 × N bytes
//   ├─────────────────────────────────────────────────────┤
//   │  Subframe 2: [AB1_L][AB1_H] [AB2_L][AB2_H] ...      │  2 × N bytes
//   └─────────────────────────────────────────────────────┘
//   where N = pixel_width × pixel_height
//         e.g. mode 6: N = 512 × 512 = 262,144 pixels  (QMP)
//              mode 0: N = 1024 × 1024 = 1,048,576 pixels (MP)
//
//  MP modes (0, 1, 4) vs QMP modes (2, 3, 5, 6):
//
//    QMP: MIPI frame = N × 5 bytes exactly (no padding).
//         pixel_width and pixel_height are read directly from adsd3100_standardModes.
//
//    MP:  MIPI frame = N × 5 + padding bytes.
//         Modes 0 and 1: MIPI = 3072 × 1707 = 5,243,904 bytes
//                        N = 1024 × 1024     = 1,048,576 pixels
//                        padding = 5,243,904 - 5,242,880 = 1,024 zero bytes
//         The kernel is launched with size = N (pixel count), so it only
//         accesses bytes [0 .. N×5 - 1].  The trailing padding bytes are
//         never read.
//
//  raw   : uint8_t*   (device) — packed input (≥ 5×N bytes)
//  depth : uint16_t*  (device) — unpacked depth plane (N elements)
//  conf  : uint16_t*  (device) — unpacked confidence plane (N elements)
//  ab    : uint16_t*  (device) — unpacked active brightness plane (N elements)
//==============================================================================
__global__
void unpack_kernel(const uint8_t* raw,
                   uint16_t* depth,
                   uint16_t* conf,
                   uint16_t* ab,
                   int width,
                   int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    if (idx >= size) return;

    // Subframe 1: Depth + Confidence interleaved at 3 bytes/pixel
    int sf1_base = idx * 3;
    depth[idx] = static_cast<uint16_t>(raw[sf1_base]
                                     | (raw[sf1_base + 1] << 8));
    conf[idx]  = static_cast<uint16_t>(raw[sf1_base + 2]);

    // Subframe 2: Active Brightness at 2 bytes/pixel, starts after subframe 1
    int sf2_base = size * 3 + idx * 2;
    ab[idx] = static_cast<uint16_t>(raw[sf2_base]
                                  | (raw[sf2_base + 1] << 8));
}


//==============================================================================
//  KERNEL: unpack_kernel
//------------------------------------------------------------------------------
//  Unpacks ADI ToF 5‑byte‑per‑pixel format:
//
//      Byte 0–1 → depth (uint16)
//      Byte 2   → confidence (upper byte only)
//      Byte 3–4 → active brightness (uint16)
//
//  raw layout per pixel:
//      [0] depth LSB
//      [1] depth MSB
//      [2] conf (8‑bit, stored in MSB position)
//      [3] ab LSB
//      [4] ab MSB
//------------------------------------------------------------------------------
//  raw   : uint8_t*   (device) — packed input
//  depth : uint16_t*  (device) — unpacked depth
//  conf  : uint16_t*  (device) — unpacked confidence
//  ab    : uint16_t*  (device) — unpacked active brightness
//==============================================================================
__global__
void unpack_kernel_2(const uint8_t* raw,
                   uint16_t* depth,
                   uint16_t* ab,
                   int width,
                   int height)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    if (idx >= size) return;

    // Subframe 1: Depth + Confidence interleaved at 3 bytes/pixel
    int sf1_base = idx * 2;
    depth[idx] = static_cast<uint16_t>(raw[sf1_base]
                                     | (raw[sf1_base + 1] << 8));
    //conf[idx]  = static_cast<uint16_t>(raw[sf1_base + 2]);

    // Subframe 2: Active Brightness at 2 bytes/pixel, starts after subframe 1
    int sf2_base = size * 2 + idx * 2;
    ab[idx] = static_cast<uint16_t>(raw[sf2_base]
                                  | (raw[sf2_base + 1] << 8));   
}


//==============================================================================
//  KERNEL: jet_kernel
//------------------------------------------------------------------------------
//  Converts depth (uint16) → RGB using a Jet colormap over
//  depth_min_mm..depth_max_mm. Closer returns clamp to the near end of the
//  colormap; invalid (0) and beyond-max depths are painted black.
//------------------------------------------------------------------------------
//  depth : uint16_t* (device)
//  rgb   : uint8_t*  (device) — output RGB image (size*3)
//==============================================================================
__global__
void jet_kernel(const uint16_t* depth,
                uint8_t* rgb,
                int size,
                float depth_min_mm,
                float depth_max_mm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    uint16_t d = depth[idx];

    uint8_t r = 0, g = 0, b = 0;
    if ((d > 0) && ((float)d <= depth_max_mm)) {
        const float scale = 255.0f / (depth_max_mm - depth_min_mm);
        int norm = min(255, max(0, (int)(((float)d - depth_min_mm) * scale)));
        r = JET_LUT[norm * 3 + 0];
        g = JET_LUT[norm * 3 + 1];
        b = JET_LUT[norm * 3 + 2];
    }

    rgb[idx * 3 + 0] = r;
    rgb[idx * 3 + 1] = g;
    rgb[idx * 3 + 2] = b;
}



//==============================================================================
//  KERNEL: depth_legend_kernel
//------------------------------------------------------------------------------
//  Draws the Jet colorbar into the depth RGB image: a vertical gradient with a
//  white outline and a white tick every tick_step_mm. Bottom of the bar is
//  depth_min_mm, top is depth_max_mm.
//------------------------------------------------------------------------------
//  rgb : uint8_t* (device) — depth RGB image, modified in place
//==============================================================================
__global__
void depth_legend_kernel(uint8_t* rgb,
                         int width,
                         int x0,
                         int y0,
                         int bar_w,
                         int bar_h,
                         float depth_min_mm,
                         float depth_max_mm,
                         float tick_step_mm)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= bar_w) || (y >= bar_h)) return;

    float fraction = 1.0f - ((float)y + 0.5f) / (float)bar_h;
    int norm = min(255, (int)(fraction * 255.0f));

    uint8_t r = JET_LUT[norm * 3 + 0];
    uint8_t g = JET_LUT[norm * 3 + 1];
    uint8_t b = JET_LUT[norm * 3 + 2];

    const float span = depth_max_mm - depth_min_mm;
    const float value = depth_min_mm + fraction * span;
    const float nearest_tick = roundf(value / tick_step_mm) * tick_step_mm;
    const float mm_per_pixel = span / (float)bar_h;

    bool outline = (x == 0) || (x == bar_w - 1) || (y == 0) || (y == bar_h - 1);
    bool tick = (nearest_tick >= depth_min_mm)
        && (fabsf(value - nearest_tick) <= mm_per_pixel)
        && (x >= bar_w - max(2, bar_w / 3));

    if (outline || tick) { r = g = b = 255; }

    int pixel = ((y0 + y) * width + (x0 + x)) * 3;
    rgb[pixel + 0] = r;
    rgb[pixel + 1] = g;
    rgb[pixel + 2] = b;
}


//==============================================================================
//  KERNEL: grayscale_kernel
//------------------------------------------------------------------------------
//  Converts 16‑bit values → grayscale RGB.
//  max_val: normalization range (use 4096 for AB, 255 for Confidence)
//------------------------------------------------------------------------------
//  input : uint16_t* (device)
//  rgb   : uint8_t*  (device)
//  max_val : float     — full-scale value mapped to 255
//  log_scale : bool    — compress the range logarithmically
//==============================================================================
__global__
void grayscale_kernel(const uint16_t* input,
                      uint8_t* rgb,
                      int size,
                      float max_val,
                      bool log_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float v = (float)input[idx];

    float scaled = log_scale
        ? (log1pf(v) / log1pf(max_val))
        : (v / max_val);

    uint8_t norm = min(255, (int)(scaled * 255.0f));

    rgb[idx * 3 + 0] = norm;
    rgb[idx * 3 + 1] = norm;
    rgb[idx * 3 + 2] = norm;
}



//==============================================================================
//  KERNEL: radial_to_z_kernel
//------------------------------------------------------------------------------
//  Converts the ADSD3500 radial depth (distance along the pixel's line of
//  sight) into Cartesian Z (distance along the optical axis).
//
//      z[i] = radial[i] * z_table[i]
//
//  z_table holds 1 / sqrt(xn^2 + yn^2 + 1) for the undistorted normalized
//  coordinates of each pixel, and 0 where no valid Z exists.
//------------------------------------------------------------------------------
//  radial  : uint16_t* (device) — radial depth in mm
//  z_table : float*    (device) — per-pixel scale
//  z       : uint16_t* (device) — Cartesian Z in mm
//==============================================================================
__global__
void radial_to_z_kernel(const uint16_t* radial,
                        const float* z_table,
                        uint16_t* z,
                        int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float value = (float)radial[idx] * z_table[idx];
    z[idx] = (uint16_t)fminf(fmaxf(value + 0.5f, 0.0f), 65535.0f);
}


//==============================================================================
//  LAUNCH WRAPPERS
//------------------------------------------------------------------------------
//  These provide clean, consistent kernel launch configuration.
//  All kernels use:
//      • 256 threads per block
//      • ceil(size / 256) blocks
//==============================================================================

//------------------------------------------------------------------------------
// Launch unpack kernel
//------------------------------------------------------------------------------
void unpack_kernel_launch(const uint8_t* raw,
                          uint16_t* depth,
                          uint16_t* conf,
                          uint16_t* ab,
                          int width,
                          int height,
                          cudaStream_t stream)
{
    int size = width * height;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    if (conf == nullptr)
    {
        unpack_kernel_2<<<blocks, threads, 0, stream>>>(
            raw, depth, ab, width, height);

    }
    else
    {
        unpack_kernel<<<blocks, threads, 0, stream>>>(
            raw, depth, conf, ab, width, height);
    }
}



//------------------------------------------------------------------------------
// Launch Jet colormap kernel
//------------------------------------------------------------------------------
void jet_kernel_launch(const uint16_t* depth,
                       uint8_t* rgb,
                       int size,
                       cudaStream_t stream,
                       float depth_min_mm,
                       float depth_max_mm)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    jet_kernel<<<blocks, threads, 0, stream>>>(
        depth, rgb, size, depth_min_mm, depth_max_mm);
}



//------------------------------------------------------------------------------
// Launch the colorbar legend kernel
//------------------------------------------------------------------------------
void depth_legend_kernel_launch(uint8_t* rgb,
                                int width,
                                int height,
                                cudaStream_t stream,
                                float depth_min_mm,
                                float depth_max_mm)
{
    int x0 = (int)(kDepthLegendBarX * width);
    int y0 = (int)(kDepthLegendBarY * height);
    int bar_w = std::min((int)(kDepthLegendBarWidth * width), width - x0);
    int bar_h = std::min((int)(kDepthLegendBarHeight * height), height - y0);
    if ((bar_w < 3) || (bar_h < 3)) return;

    dim3 threads(16, 16);
    dim3 blocks((bar_w + threads.x - 1) / threads.x,
        (bar_h + threads.y - 1) / threads.y);

    depth_legend_kernel<<<blocks, threads, 0, stream>>>(
        rgb, width, x0, y0, bar_w, bar_h, depth_min_mm, depth_max_mm,
        depth_legend_tick_step_mm(depth_min_mm, depth_max_mm));
}


//------------------------------------------------------------------------------
// Launch grayscale kernel
//  max_val: 4096 for Active Brightness, 255 for Confidence
//------------------------------------------------------------------------------
void grayscale_kernel_launch(const uint16_t* input,
                             uint8_t* rgb,
                             int size,
                             cudaStream_t stream,
                             float max_val,
                             bool log_scale)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    grayscale_kernel<<<blocks, threads, 0, stream>>>(
        input, rgb, size, max_val, log_scale);
}



//------------------------------------------------------------------------------
// Launch radial → Cartesian Z kernel
//------------------------------------------------------------------------------
void radial_to_z_kernel_launch(const uint16_t* radial,
                               const float* z_table,
                               uint16_t* z,
                               int size,
                               cudaStream_t stream)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    radial_to_z_kernel<<<blocks, threads, 0, stream>>>(
        radial, z_table, z, size);
}



//------------------------------------------------------------------------------
// Launch shift‑and‑cast kernel
//------------------------------------------------------------------------------
void shift_and_cast_kernel(const uint16_t* in,
                           uint8_t* out,
                           int count,
                           cudaStream_t stream)
{
    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    shift_and_cast_kernel<<<blocks, threads, 0, stream>>>(
        in, out, count);
}

