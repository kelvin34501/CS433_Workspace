#include <conv_cpu.h>

void conv_cpu(float* res, float* inp, float* kernel, int N, int C, int H, int W,
              int F, int KX, int KY) {
    int H_ = H - KX + 1;
    int W_ = W - KY + 1;
    for (int n = 0; n < N; ++n) {
        for (int f = 0; f < F; ++f) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H_; ++h) {
                    for (int w = 0; w < W_; ++w) {
                        for (int i = 0; i < KX; ++i) {
                            for (int j = 0; j < KY; ++j) {
                                // res[n,f,h,w]
                                // inp[n,c,h+i,w+j]
                                // kernel[f,c,i,j]
                                res[((n * F + f) * H_ + h) * W_ + w] +=
                                    inp[((n * C + c) * H + h + i) * W + w + j] *
                                    kernel[((f * C + c) * KX + i) * KY + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

void init_tensor(float* mat, int a, int b, int c, int d) {
    constexpr int base = 10;
    for (int i = 0; i < a * b * c * d; i++) {
        mat[i] = i % base;
    }
}