#ifndef RBE_FRAME_API_H
#define RBE_FRAME_API_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// All functions return 0 on success, -1 on invalid descriptor/aliasing.
// Codes are 15-bit XOR address, 15-bit parity mask, 1-bit sign; bit31 reserved.
int rbe_frame_compose(uint32_t left, uint32_t right, uint32_t *out);
int rbe_frame_transpose(uint32_t code, uint32_t *out);
int rbe_frame_fuse(const uint32_t *codes, size_t length, uint32_t *out);
int rbe_route32(const uint32_t *src, uint32_t *dst, size_t words,
                uint32_t log2_n, uint32_t code);
int rbe_route64(const uint64_t *src, uint64_t *dst, size_t words,
                uint32_t log2_n, uint32_t code);
#ifdef __cplusplus
}
#endif
#endif
