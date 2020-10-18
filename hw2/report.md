# Report

## Problem 1

## Problem 2

## Result

First compare different methods: (block size = 32)
| Name                           | Time (ns)   |
| ------------------------------ | ----------- |
| CPU                            | 16532017303 |
| GPU (direct convolution)       | 704137263   |
| GPU (im2col + matmul)          | 351770067   |
| GPU (im2col + matmul + tiling) | 158283863   |

Then compare different tile size
| Tile Size | Time (ns)  |
| --------- | ---------- |
| 2         | 1174414726 |
| 4         | 221860553  |
| 8         | 71264879   |
| 16        | 109263822  |
| 32        | 158283863  |
| 64 *      | 43435916   |

The tile