#include <conv_gpu.h>
#include <cuda_runtime.h>

__global__ void cuda_conv_gpu(float* res, float* inp, float* kernel, int N,
                              int C, int H, int W, int F, int KX, int KY) {
    // there will be no optimization!
    int H_ = H - KX + 1;
    int W_ = W - KY + 1;

    // figure out which number need to compute
    int n = blockIdx.x; // n
    int f = blockIdx.y; // f
    int h_beg = threadIdx.x;
    int w_beg = threadIdx.y;

    // we need to iterate over h and w
    for (int h = h_beg; h < H_; h += blockDim.x) {
        // we need to iterate over F
        for (int w = w_beg; w < W_; w += blockDim.y) {
            // we need to compute the number within the block
            // that is, a number on h and w
            // complete coordinate: res[n, f, h, w]
            // to do this, we need C * 3 * 3 inner iterations
            // we need access to data
            //      inp[n, c, h+i, w+j]
            //      ker[f, c, i, j]
            float curr = 0;
            // order of summation should not affect result? check this
            for (int c = 0; c < C; c++) { // c++
                for (int i = 0; i < KX; i++) {
                    for (int j = 0; j < KY; j++) {
                        curr += inp[((n * C + c) * H + h + i) * W + w + j] *
                                kernel[((f * C + c) * KX + i) * KY + j];
                    }
                }
            }
            res[((n * F + f) * H_ + h) * W_ + w] += curr; // should do this here only once
        }
    }
}

// grid_size n * f
// block_size 32 * 32
void conv_gpu(float* res, float* inp, float* kernel, int N, int C, int H, int W,
              int F, int KX, int KY) {
    // pass memory to cuda device
    int H_ = H - KX + 1;
    int W_ = W - KY + 1;

    float* cuda_res{};
    float* cuda_inp{};
    float* cuda_kernel{};
    size_t size_res = N * F * H_ * W_;
    size_t size_inp = N * C * H * W;
    size_t size_kernel = F * C * KX * KY;
    cudaMalloc(&cuda_res, size_res * sizeof(float));
    cudaMalloc(&cuda_inp, size_inp * sizeof(float));
    cudaMalloc(&cuda_kernel, size_kernel * sizeof(float));
    cudaMemcpy(cuda_inp, inp, size_inp * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_kernel, kernel, size_kernel * sizeof(float),
               cudaMemcpyHostToDevice);

    // call cuda kernel
    // first figure out how much block is needed
    // each block is 32 threads
    // each grid covers one in N and one in F.
    int num_block_x = N;
    int num_block_y = F;
    dim3 num_blocks(num_block_x, num_block_y);
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    cuda_conv_gpu<<<num_blocks, block_size>>>(cuda_res, cuda_inp, cuda_kernel,
                                              N, C, H, W, F, KX, KY);

    // pass result back to cpu
    cudaMemcpy(res, cuda_res, size_res * sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(cuda_res);
    cudaFree(cuda_inp);
    cudaFree(cuda_kernel);
}
