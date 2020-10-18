#ifndef CONV_GPU_H
#define CONV_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

extern void conv_gpu(float* res, float* inp, float* kernel, int N, int C, int H,
                     int W, int F, int KX, int KY);
constexpr int BLOCK_SIZE_X = 32;
constexpr int BLOCK_SIZE_Y = 32;

#ifdef __cplusplus
}
#endif

#endif