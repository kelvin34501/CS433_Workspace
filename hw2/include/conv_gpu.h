#ifndef CONV_GPU_H
#define CONV_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

extern void conv_gpu(float* res, float* inp, float* kernel, int N, int C, int H,
                     int W, int F, int KX, int KY);
extern void conv_gpu_2(float* res, float* inp, float* kernel, int N, int C,
                       int H, int W, int F, int KX, int KY);
extern void conv_gpu_3(float* res, float* inp, float* kernel, int N, int C,
                       int H, int W, int F, int KX, int KY);
constexpr int BLOCK_SIZE_X = 4;
constexpr int BLOCK_SIZE_Y = 4;
constexpr int TILE_SIZE = 4;

#ifdef __cplusplus
}
#endif

#endif