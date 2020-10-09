# HW1

## build instruction

Default build type is ```RelWithDebInfo```. For optimized result, you should use ```Release```.
To build:
```bash
mkdir build && cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=$YOUR_BUILD_TYPE
ninja
```

To see the results of unit tests: (make sure you are in ```build``` directory)
```bash
./hw1p1_test
./hw1p2_test
```

To see the results of benchmarks: (make sure you are in ```build``` directory)
```bash
./hw1p1_benchmark
./hw1p2_benchmark
```

## Test results

+ hw1p1_test

```
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from HW1P1
[ RUN      ] HW1P1.SERIAL
[       OK ] HW1P1.SERIAL (0 ms)
[ RUN      ] HW1P1.PAR
[       OK ] HW1P1.PAR (0 ms)
[ RUN      ] HW1P1.PAR_LOCK
[       OK ] HW1P1.PAR_LOCK (0 ms)
[ RUN      ] HW1P1.PAR_RED
[       OK ] HW1P1.PAR_RED (0 ms)
[ RUN      ] HW1P1.PAR_RED_NEQ
[       OK ] HW1P1.PAR_RED_NEQ (0 ms)
[----------] 5 tests from HW1P1 (0 ms total)

[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (0 ms total)
[  PASSED  ] 5 tests.
```
+ hw1p2_test

```
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from HW1P2
[ RUN      ] HW1P2.SERIAL
[       OK ] HW1P2.SERIAL (0 ms)
[ RUN      ] HW1P2.SERIAL_ALT
[       OK ] HW1P2.SERIAL_ALT (0 ms)
[ RUN      ] HW1P2.PAR
[       OK ] HW1P2.PAR (0 ms)
[ RUN      ] HW1P2.PAR_ALT
[       OK ] HW1P2.PAR_ALT (0 ms)
[----------] 4 tests from HW1P2 (0 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test suite ran. (0 ms total)
[  PASSED  ] 4 tests.
```

## Benchmark results

+ hw1p1_benchmark

```
020-10-09T22:59:12+08:00
Running ./hw1p1_benchmark
Run on (8 X 3900 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 256 KiB (x4)
  L3 Unified 6144 KiB (x1)
Load Average: 1.57, 1.41, 1.04
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
benchmark_generic_sum               6069566 ns      6069529 ns          118
benchmark_generic_sum_omp/1       151786012 ns    151734272 ns            5
benchmark_generic_sum_omp/2       264594331 ns    264590370 ns            3
benchmark_generic_sum_omp/3       437720614 ns    437386862 ns            2
benchmark_generic_sum_omp/4       565434925 ns    565278766 ns            1
benchmark_generic_sum_omp/5       572355642 ns    572231647 ns            1
benchmark_generic_sum_omp/6       655638027 ns    655147371 ns            1
benchmark_generic_sum_omp/7       711132320 ns    711124631 ns            1
benchmark_generic_sum_omp/8       700406198 ns    700344057 ns            1
benchmark_generic_sum               8407318 ns      8407100 ns          120
benchmark_generic_sum_omp_lock/1  152801489 ns    152800635 ns            4
benchmark_generic_sum_omp_lock/2  587645023 ns    587644701 ns            1
benchmark_generic_sum_omp_lock/3  855132956 ns    854095879 ns            1
benchmark_generic_sum_omp_lock/4 1146710500 ns   1127109708 ns            1
benchmark_generic_sum_omp_lock/5 1059183796 ns    721830126 ns            1
benchmark_generic_sum_omp_lock/6 1261012615 ns    998023706 ns            1
benchmark_generic_sum_omp_lock/7 1282394625 ns    975059470 ns            1
benchmark_generic_sum_omp_lock/8 1371125777 ns    959406516 ns            1
benchmark_generic_sum               5998312 ns      5998279 ns          118
benchmark_generic_sum_omp_red/1     4275284 ns      4275260 ns          166
benchmark_generic_sum_omp_red/2     2202765 ns      2202751 ns          315
benchmark_generic_sum_omp_red/3     2264638 ns      2263392 ns          327
benchmark_generic_sum_omp_red/4     1702365 ns      1702350 ns          386
benchmark_generic_sum_omp_red/5     1510967 ns      1510967 ns          448
benchmark_generic_sum_omp_red/6     1418715 ns      1418538 ns          445
benchmark_generic_sum_omp_red/7     1346728 ns      1346651 ns          534
benchmark_generic_sum_omp_red/8     1369365 ns      1369356 ns          527
benchmark_generic_sum               5887914 ns      5887084 ns          122
```

+ hw1p2_benchmark

```
2020-10-09T23:00:21+08:00
Running ./hw1p2_benchmark
Run on (8 X 3900 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 256 KiB (x4)
  L3 Unified 6144 KiB (x1)
Load Average: 1.91, 1.61, 1.14
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
benchmark_matmul             61474194 ns     61466916 ns           11
benchmark_matmul_par/1       60704807 ns     60704618 ns           11
benchmark_matmul_par/2       35812748 ns     35811639 ns           22
benchmark_matmul_par/3       26152387 ns     26152156 ns           23
benchmark_matmul_par/4       22425261 ns     22293886 ns           33
benchmark_matmul_par/5       17688918 ns     17681518 ns           38
benchmark_matmul_par/6       12792128 ns     12792143 ns           48
benchmark_matmul_par/7       11508200 ns     11508119 ns           59
benchmark_matmul_par/8       10764284 ns     10703919 ns           62
benchmark_matmul_alt         63734428 ns     63734019 ns           11
benchmark_matmul_par_alt/1   59747172 ns     59746714 ns           11
benchmark_matmul_par_alt/2   30933082 ns     30932980 ns           23
benchmark_matmul_par_alt/3   20594883 ns     20594899 ns           34
benchmark_matmul_par_alt/4   15781729 ns     15781730 ns           44
benchmark_matmul_par_alt/5   13879264 ns     13879109 ns           51
benchmark_matmul_par_alt/6   11827001 ns     11826934 ns           59
benchmark_matmul_par_alt/7   10270290 ns     10270217 ns           63
benchmark_matmul_par_alt/8    9429956 ns      9429854 ns           72
```