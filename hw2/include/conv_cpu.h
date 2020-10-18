#ifndef CONV_CPU_H
#define CONV_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

extern void conv_cpu(float* res, float* inp, float* kernel, int N, int C, int H,
                     int W, int F, int KX, int KY);
extern void init_tensor(float* mat, int a, int b, int c, int d);

#ifdef __cplusplus
}
#endif

#endif