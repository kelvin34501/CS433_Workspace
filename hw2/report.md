# Report

## Optimization Method

The main idea is to use ```im2col``` that expand the feature map into a larger matrix, and then the convolution can be represented by the kernel multiplying the expanded feature map. The reason of useing ```im2col``` boost performance of convolution might be:
+ Matrix Multiplication can be easier to parallelize.
+ Matrix Multiplication can be more friendly to cache in GPU stream multiprocessors.
+ Matrix Multiplication can be optimized through shared memory (that is, tiling) so that the memory bandwidth can be greatly saved.

The steps are listed as follows:
1. Allocate a piece of GPU Memory. Initialize it with all 0s.
2. Use a kernel does ```im2col``` and other work, like transposing the matrix and padding them to tile size. After doing this we can use matrix multiplication to do convolution.
3. Use a kernel that multiplies 2 matrices to get the convolution result.

## Result

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
| 64 *      | 43435916   |

\* : actually this is not feasible, as one stream multiprocessor only has 2048 threads, and device param says one thread block can only hold 1024 threads. Do not know why the program compiles. No matter what makes it compile, the time recording here doesn't have a meaning.

When the tile size is 8, we achieve best performance.

3. Finally we check the effect of unrolling inner loop in matrix multiplication.

| Name         | Time (ns) |
| ------------ | --------- |
| not unrolled | 158283863 |
| unrolled     | 310164293 |

# Analysis

From the results