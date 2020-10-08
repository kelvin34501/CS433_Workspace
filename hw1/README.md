# HW1

## Test results

+ hw1p1_test

```
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from HW1P1
[ RUN      ] HW1P1.SERIAL
[       OK ] HW1P1.SERIAL (0 ms)
[ RUN      ] HW1P1.PAR
[       OK ] HW1P1.PAR (0 ms)
[ RUN      ] HW1P1.PAR_LOCK
[       OK ] HW1P1.PAR_LOCK (0 ms)
[ RUN      ] HW1P1.PAR_RED
[       OK ] HW1P1.PAR_RED (0 ms)
[----------] 4 tests from HW1P1 (0 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test suite ran. (0 ms total)
[  PASSED  ] 4 tests.
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
2020-10-08T22:56:07+08:00
Running ./hw1p1_benchmark
Run on (8 X 3900 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 256 KiB (x4)
  L3 Unified 6144 KiB (x1)
Load Average: 1.60, 1.62, 1.81
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
benchmark_generic_sum               6673255 ns      6671956 ns          120
benchmark_generic_sum_omp/1       135448227 ns    135400281 ns            5
benchmark_generic_sum_omp/2       250182136 ns    250154979 ns            3
benchmark_generic_sum_omp/3       379374515 ns    363215715 ns            2
benchmark_generic_sum_omp/4       540970096 ns    505865858 ns            1
benchmark_generic_sum_omp/5       580141131 ns    580071832 ns            1
benchmark_generic_sum_omp/6       690479179 ns    676277554 ns            1
benchmark_generic_sum_omp/7       710543007 ns    707805389 ns            1
benchmark_generic_sum_omp/8       757728235 ns    620699082 ns            1
benchmark_generic_sum               6433511 ns      6432838 ns           85
benchmark_generic_sum_omp_lock/1  137609074 ns    137591901 ns            5
benchmark_generic_sum_omp_lock/2  290901075 ns    289255520 ns            3
benchmark_generic_sum_omp_lock/3  406722405 ns    375589024 ns            2
benchmark_generic_sum_omp_lock/4  585884280 ns    585884278 ns            1
benchmark_generic_sum_omp_lock/5  650339337 ns    649791054 ns            1
benchmark_generic_sum_omp_lock/6  767792785 ns    767786527 ns            1
benchmark_generic_sum_omp_lock/7  867933388 ns    839722038 ns            1
benchmark_generic_sum_omp_lock/8  928890898 ns    928772276 ns            1
benchmark_generic_sum               6208493 ns      6206852 ns          121
benchmark_generic_sum_omp_red/1     9181121 ns      9179414 ns           83
benchmark_generic_sum_omp_red/2     4941681 ns      4940430 ns          147
benchmark_generic_sum_omp_red/3     3388182 ns      3387237 ns          205
benchmark_generic_sum_omp_red/4     2558615 ns      2557938 ns          244
benchmark_generic_sum_omp_red/5     2667542 ns      2666676 ns          288
benchmark_generic_sum_omp_red/6     2328488 ns      2325557 ns          334
benchmark_generic_sum_omp_red/7     2275485 ns      2234399 ns          377
benchmark_generic_sum_omp_red/8     2008220 ns      1932709 ns          381
```

+ hw1p2_benchmark

```
2020-10-08T22:58:23+08:00
Running ./hw1p2_benchmark
Run on (8 X 3900 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 256 KiB (x4)
  L3 Unified 6144 KiB (x1)
Load Average: 1.92, 1.79, 1.85
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
benchmark_matmul             64290877 ns     64284564 ns           10
benchmark_matmul_par/1       66718295 ns     66704186 ns           12
benchmark_matmul_par/2       33495025 ns     33491064 ns           20
benchmark_matmul_par/3       22071813 ns     22069439 ns           29
benchmark_matmul_par/4       18306723 ns     18303620 ns           42
benchmark_matmul_par/5       16658261 ns     16558858 ns           46
benchmark_matmul_par/6       15396404 ns     15089624 ns           52
benchmark_matmul_par/7       12778783 ns     12296192 ns           45
benchmark_matmul_par/8       12528255 ns     12017053 ns           44
benchmark_matmul_alt         63331130 ns     63320719 ns           10
benchmark_matmul_par_alt/1   65062229 ns     65052553 ns           11
benchmark_matmul_par_alt/2   33009669 ns     33006697 ns           22
benchmark_matmul_par_alt/3   22671961 ns     22666807 ns           34
benchmark_matmul_par_alt/4   17288567 ns     17284405 ns           43
benchmark_matmul_par_alt/5   14811066 ns     14800129 ns           50
benchmark_matmul_par_alt/6   12694588 ns     12549452 ns           57
benchmark_matmul_par_alt/7   11613924 ns     11174800 ns           46
benchmark_matmul_par_alt/8   11022873 ns     10344988 ns           70
```