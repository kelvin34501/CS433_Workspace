#include <conv_gpu.h>
#include <cuda_runtime.h>
#include <iostream>

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
            res[((n * F + f) * H_ + h) * W_ + w] += curr;
            // should do this here only once
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

// im2col
// actually, itis im2row and then flatten input channel
// and then tranpose (why is it so complicated?)
// inp[N, C, H, W] => inp'[N, C * KX * KY, H_ * W_]
// implementation
// inp[N, C, H, W] => [N, C, KX*KY, H_*W_] => inp'[N, C * KX * KY, H_ * W_]
// no work need to be done at last step
__global__ void input_im2col_flatten_transpose(float* res, float* inp, int N,
                                               int C, int H, int W, int KX,
                                               int KY) {
    int H_ = H - KX + 1;
    int W_ = W - KY + 1;

    // figure out which value we need to compute
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h_beg = threadIdx.x;
    int w_beg = threadIdx.y;
    for (int i = 0; i < KX; i++) {
        for (int j = 0; j < KY; j++) {
            for (int h = h_beg; h < H_; h += blockDim.x) {
                for (int w = w_beg; w < W_; w += blockDim.y) {
                    // res[n, c, i * KY + j, h * W_ + w]
                    // inp[n, c, h+i, w+j]
                    res[((((n * C + c) * KX + i) * KY + j) * H_ + h) * W_ + w] =
                        inp[((n * C + c) * H + h + i) * W + w + j];
                }
            }
        }
    }
}

// deal with kernel
// kern[F, C, KX, KY] => kern'[F, C * KX * KY]
// these two should be the same (in row major)

// matmul
// kern'[F, C * KX * KY] @ inp'[N, C * KX * KY, H_ * W_] // lets ignore N
// res'[N, F, H_ * W_]
// which will be the same as
// res''[N, F, H_, W_]
__global__ void matmul_naive(float* res, float* inp_derive,
                             float* kernel_derive, int N, int C, int H, int W,
                             int F, int KX, int KY) {
    int H_ = H - KX + 1;
    int W_ = W - KY + 1;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // figure out position
    int f = bx * blockDim.x + tx;  // less than F
    int hw = by * blockDim.y + ty; // less than H_ * W_
    int width = C * KX * KY;

    if (f >= F || hw >= H_ * W_) {
        return;
    }

    // iterate over n
    for (int n = 0; n < N; n++) {
        float p_value = 0.0f;
        for (int k = 0; k < width; k++) {
            // kernel[f, k] * inp_derive[n, k, hw]
            p_value += kernel_derive[f * width + k] *
                       inp_derive[n * width * H_ * W_ + k * H_ * W_ + hw];
        }
        // load into [n, f, hw]
        res[n * F * H_ * W_ + f * H_ * W_ + hw] = p_value;
    }
}

__global__ void clear(float* res, int N, int padding_width, int padding_H_W_) {
    int n = blockIdx.x;
    int a_beg = threadIdx.x;
    int b_beg = threadIdx.y;
    for (int a = a_beg; a < padding_width; a += blockDim.x) {
        for (int b = b_beg; b < padding_H_W_; b += blockDim.y) {
            res[n * padding_width * padding_H_W_ + a * padding_H_W_ + b] = 0.0f;
        }
    }
}

// with padding
// im2col
// actually, itis im2row and then flatten input channel
// and then tranpose (why is it so complicated?)
// inp[N, C, H, W] => inp'[N, C * KX * KY, H_ * W_]
// implementation
// inp[N, C, H, W] => [N, C, KX*KY, H_*W_] => inp'[N, C * KX * KY, H_ * W_]
// no work need to be done at last step
__global__ void input_im2col_flatten_transpose_padding(float* res, float* inp,
                                                       int N, int C, int H,
                                                       int W, int KX, int KY,
                                                       int padding_width,
                                                       int padding_H_W_) {
    int H_ = H - KX + 1;
    int W_ = W - KY + 1;

    // figure out which value we need to compute
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h_beg = threadIdx.x;
    int w_beg = threadIdx.y;
    for (int i = 0; i < KX; i++) {
        for (int j = 0; j < KY; j++) {
            for (int h = h_beg; h < H_; h += blockDim.x) {
                for (int w = w_beg; w < W_; w += blockDim.y) {
                    // res[n, c, i * KY + j, h * W_ + w]
                    // inp[n, c, h+i, w+j]
                    res[n * padding_width * padding_H_W_ +
                        ((c * KX + i) * KY + j) * padding_H_W_ + h * W_ + w] =
                        inp[((n * C + c) * H + h + i) * W + w + j];
                }
            }
        }
    }
}

// matmul
// kern'[F, C * KX * KY] @ inp'[N, C * KX * KY, H_ * W_] // lets ignore N
// res'[N, F, H_ * W_]
// which will be the same as
// res''[N, F, H_, W_]
// implementation note: BLOCK_SIZE_X == BLOCK_SIZE_Y == TILE_SIZE
// otherwise malfunction
__global__ void matmul_tile(float* res, float* inp_derive, float* kernel_derive,
                            int N, int C, int H, int W, int F, int KX, int KY,
                            int padding_width, int padding_H_W_) {
    int H_ = H - KX + 1;
    int W_ = W - KY + 1;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int n = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // figure out position
    int f = bx * blockDim.x + tx;  // less than F
    int hw = by * blockDim.y + ty; // less than H_ * W_

    // collaborative loading
    float p_value = 0.0f;
    // compute the m-th tile
    for (int m = 0; m < padding_width / TILE_SIZE; m++) {
        // define shared memory
        __shared__ float inp_block_shared[BLOCK_SIZE_X][TILE_SIZE];
        __shared__ float kern_block_shared[BLOCK_SIZE_Y][TILE_SIZE];

        // load kernel_derive[f, m * TILE_SIZE + ty]
        kern_block_shared[tx][ty] =
            kernel_derive[f * padding_width + m * TILE_SIZE + ty];
        // load inp_derive[n, m * TILE_SIZE + tx, hw]
        inp_block_shared[ty][tx] =
            inp_derive[n * padding_width * padding_H_W_ +
                       (m * TILE_SIZE + tx) * padding_H_W_ + hw];
        __syncthreads();

        // sum within the block
#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            p_value += kern_block_shared[tx][k] * inp_block_shared[ty][k];
        }
        __syncthreads();
    }
    // load result into res[n, f, hw]
    if (f >= F || hw >= H_ * W_) {
        return;
    } else {
        res[n * F * H_ * W_ + f * H_ * W_ + hw] = p_value;
    }
}

// conv_gpu_2
void conv_gpu_2(float* res, float* inp, float* kernel, int N, int C, int H,
                int W, int F, int KX, int KY) {
    // pass memory to cuda device
    int H_ = H - KX + 1;
    int W_ = W - KY + 1;

    float* cuda_res{};
    float* cuda_inp{};
    float* cuda_kernel{};
    float* cuda_inp_derive{};
    size_t size_res = N * F * H_ * W_;
    size_t size_inp = N * C * H * W;
    size_t size_kernel = F * C * KX * KY;
    size_t size_inp_derive = N * H_ * W_ * C * KX * KY;
    cudaMalloc(&cuda_res, size_res * sizeof(float));
    cudaMalloc(&cuda_inp, size_inp * sizeof(float));
    cudaMalloc(&cuda_kernel, size_kernel * sizeof(float));
    cudaMalloc(&cuda_inp_derive, size_inp_derive * sizeof(float));
    cudaMemcpy(cuda_inp, inp, size_inp * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_kernel, kernel, size_kernel * sizeof(float),
               cudaMemcpyHostToDevice);

    // call cuda kernel
    dim3 num_blocks(N, C);
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    input_im2col_flatten_transpose<<<num_blocks, block_size>>>(
        cuda_inp_derive, cuda_inp, N, C, H, W, KX, KY);
    dim3 num_blocks2((F + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                     ((H_ * W_) + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_size2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    matmul_naive<<<num_blocks2, block_size2>>>(
        cuda_res, cuda_inp_derive, cuda_kernel, N, C, H, W, F, KX, KY);

    // pass result back to cpu
    cudaMemcpy(res, cuda_res, size_res * sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(cuda_res);
    cudaFree(cuda_inp);
    cudaFree(cuda_inp_derive);
    cudaFree(cuda_kernel);
}

// conv_gpu_3
void conv_gpu_3(float* res, float* inp, float* kernel, int N, int C, int H,
                int W, int F, int KX, int KY) {
    // pass memory to cuda device
    int H_ = H - KX + 1;
    int W_ = W - KY + 1;

    float* cuda_res{};
    float* cuda_inp{};
    float* cuda_kernel{};
    float* cuda_inp_derive{};
    size_t size_res = N * F * H_ * W_;
    size_t size_inp = N * C * H * W;
    size_t size_kernel = F * C * KX * KY;
    size_t width = C * KX * KY;
    size_t padded_width = ((width + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    size_t padded_H_W_ = ((H_ * W_ + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    size_t size_inp_derive = N * padded_width * padded_H_W_;
    cudaMalloc(&cuda_res, size_res * sizeof(float));
    cudaMalloc(&cuda_inp, size_inp * sizeof(float));
    cudaMalloc(&cuda_kernel, size_kernel * sizeof(float));
    cudaMalloc(&cuda_inp_derive, size_inp_derive * sizeof(float));
    cudaMemcpy(cuda_inp, inp, size_inp * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_kernel, kernel, size_kernel * sizeof(float),
               cudaMemcpyHostToDevice);

    // call cuda kernel
    dim3 block_size0(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    clear<<<N, block_size0>>>(cuda_inp_derive, N, padded_width, padded_H_W_);
    dim3 num_blocks1(N, C);
    dim3 block_size1(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    input_im2col_flatten_transpose_padding<<<num_blocks1, block_size1>>>(
        cuda_inp_derive, cuda_inp, N, C, H, W, KX, KY, padded_width,
        padded_H_W_);
    dim3 num_blocks2(F / BLOCK_SIZE_X, padded_H_W_ / BLOCK_SIZE_Y, N);
    dim3 block_size2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    matmul_tile<<<num_blocks2, block_size2>>>(cuda_res, cuda_inp_derive,
                                              cuda_kernel, N, C, H, W, F, KX,
                                              KY, padded_width, padded_H_W_);

    // pass result back to cpu
    cudaMemcpy(res, cuda_res, size_res * sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(cuda_res);
    cudaFree(cuda_inp);
    cudaFree(cuda_inp_derive);
    cudaFree(cuda_kernel);
}
