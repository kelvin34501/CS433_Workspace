# Report

## Optimization Method

The main idea is to use ```im2col``` that expand the feature map into a larger matrix, and then the convolution can be formulated by the kernel multiplying the expanded feature map. The reason of using ```im2col``` boost performance of convolution might be:
+ Matrix Multiplication can be easier to parallelize.
+ Matrix Multiplication can be more friendly to cache in GPU stream multiprocessors.
+ Matrix Multiplication can be optimized through shared memory (that is, tiling) so that the memory bandwidth can be greatly saved.

The steps are listed as follows:
1. Allocate a piece of GPU Memory. Initialize it with all 0s.
2. Use a kernel does ```im2col``` and other work, like transposing the matrix and padding them to tile size. After doing this we can use matrix multiplication to do convolution.
3. Use a kernel that multiplies 2 matrices to get the convolution result.

## Result

All GPU time records include the time copying memory from CPU to GPU.

1. First compare different methods: (block size = 32)

| Name                           | Time (ns)   |
| ------------------------------ | ----------- |
| CPU                            | 16532017303 |
| GPU (direct convolution)       | 704137263   |
| GPU (im2col + matmul)          | 351770067   |
| GPU (im2col + matmul + tiling) | 158283863   |

Results shows that use ```im2col``` and then transform convolution into matrix multiplication does improve the performance to a great extent.

2. Then compare different tile size

| Tile Size | Time (ns)  |
| --------- | ---------- |
| 2         | 1174414726 |
| 4         | 221860553  |
| 8         | 71264879   |
| 16        | 109263822  |
| 32        | 158283863  |

When the tile size is 8, we achieve best performance (accelaration ratio 232 compared to CPU).

3. Finally we check the effect of unrolling inner loop in matrix multiplication. (block size = 32)

This is done by adding one line 
```cpp 
#pragma unroll
```
before the ```for``` of the inner loop.

| Name         | Time (ns) |
| ------------ | --------- |
| not unrolled | 158283863 |
| unrolled     | 154889926 |

# Analysis

From the results we have some interesting facts and we will try to analyze them here.
+ Tile Size 8 gives best performance.
    + Too small tile size cannot fully utilize stream multiprocessor.

    One stream multiprocessor can execute 32 threads at a time. When tile size is 2 or 4, there will be 28 or 16 idle processors, and the computation power is not fully utilized.

    + Too small tile size cannot shadow the latency of memory access.
        
    Accessing memory takes large amounts of cycles on GPU, same with the case on CPU. GPU uses multiple warps in one thread blocks to shadow the latency of memory loading, through switching to another warp when one warp suspends for memory access. For tile size like 2 or 4, there will be only 4 or 16 threads running, thus the memory access will not be shadowed, and the performance will not be ideal.
    
    + Too large tile size affect the level of parallelism. 
    
    One stream multiprocessor can run 32 threads at one particular moment. For tile size like 32, we will have 1024 threads in one block and assigned to one stream multiprocessor. Although larger tile size saves more memory readings, it means there will be more warps waiting for stream multiprocessor, leading to less level of parallelism.
+ Loop Unrolling doesn't seem to be effective.
    + Benchmarking Environment may be noisy. The public server for instance, may have more than 1 cuda kernel running on the graphics card, thus grealy influence the credit of benchmarking results.
    + Option #1: Compiler choose not to unroll the loop for other obscure reasons. 
    + Option #2: Compiler unroll the loop in both cases, explicity requested or not.
