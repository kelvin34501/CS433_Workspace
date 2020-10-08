# HW1

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
2020-10-09T00:34:14+08:00
Running ./hw1p1_benchmark
Run on (8 X 3900 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 256 KiB (x4)
  L3 Unified 6144 KiB (x1)
Load Average: 2.65, 4.64, 5.17
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
benchmark_generic_sum                  6671777 ns      6671748 ns          100
benchmark_generic_sum_omp/1          138911231 ns    137220107 ns            5
benchmark_generic_sum_omp/2          281775944 ns    277226358 ns            3
benchmark_generic_sum_omp/3          394479586 ns    349686196 ns            2
benchmark_generic_sum_omp/4          550630514 ns    494701924 ns            2
benchmark_generic_sum_omp/5          625707154 ns    624710725 ns            1
benchmark_generic_sum_omp/6          737489036 ns    737444030 ns            1
benchmark_generic_sum_omp/7          714175991 ns    672292390 ns            1
benchmark_generic_sum_omp/8          806230698 ns    762239600 ns            1
benchmark_generic_sum                  6497550 ns      6497527 ns          115
benchmark_generic_sum_omp_lock/1     142813372 ns    142699370 ns            5
benchmark_generic_sum_omp_lock/2     446440080 ns    446414660 ns            2
benchmark_generic_sum_omp_lock/3     677415263 ns    574478017 ns            1
benchmark_generic_sum_omp_lock/4     988452864 ns    895368888 ns            1
benchmark_generic_sum_omp_lock/5    1110937111 ns   1081705195 ns            1
benchmark_generic_sum_omp_lock/6    1197095101 ns   1191632717 ns            1
benchmark_generic_sum_omp_lock/7    1355005623 ns   1229417474 ns            1
benchmark_generic_sum_omp_lock/8    1477722004 ns   1346610774 ns            1
benchmark_generic_sum                 10895860 ns     10889315 ns           48
benchmark_generic_sum_omp_red/1        9219981 ns      9209312 ns          100
benchmark_generic_sum_omp_red/2        2894151 ns      2893890 ns          253
benchmark_generic_sum_omp_red/3        1948286 ns      1948275 ns          409
benchmark_generic_sum_omp_red/4        1825490 ns      1825491 ns          418
benchmark_generic_sum_omp_red/5        1829143 ns      1828741 ns          452
benchmark_generic_sum_omp_red/6        1738110 ns      1738092 ns          414
benchmark_generic_sum_omp_red/7        1906961 ns      1749298 ns          390
benchmark_generic_sum_omp_red/8        1822909 ns      1760146 ns          311
benchmark_generic_sum                  7284591 ns      7284345 ns          100
benchmark_generic_sum_omp_red_neq/1   10259419 ns     10258838 ns           78
benchmark_generic_sum_omp_red_neq/2    5306878 ns      5306262 ns          100
benchmark_generic_sum_omp_red_neq/3    3425248 ns      3425188 ns          206
benchmark_generic_sum_omp_red_neq/4    2624202 ns      2624109 ns          290
benchmark_generic_sum_omp_red_neq/5    2676498 ns      2676476 ns          281
benchmark_generic_sum_omp_red_neq/6    2383722 ns      2331041 ns          327
benchmark_generic_sum_omp_red_neq/7    2122274 ns      2057006 ns          365
benchmark_generic_sum_omp_red_neq/8    1978956 ns      1960295 ns          397
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