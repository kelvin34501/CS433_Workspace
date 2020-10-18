#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel function to add elements of two arrays on gpu
__global__ void add(int n, float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    printf("%d, %d, %d\n", blockIdx.x, blockDim.x, gridDim.x);
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    constexpr int N = 1 << 20;
    constexpr int block_size = 256;
    // allocate unified memory -- accessible from CPU or GPU
    float* x{};
    float* y{};
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run kernel on 1M elements on cpu
    int num_blocks = (N + block_size - 1) / block_size;
    add<<<num_blocks, block_size>>>(N, x, y);

    // wait for gpu to finish before accessing the host
    cudaDeviceSynchronize();

    // check for errors
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        max_error = fmax(max_error, fabs(y[i] - 3.0f));
    }
    std::cout << "max error: " << max_error << '\n';

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}