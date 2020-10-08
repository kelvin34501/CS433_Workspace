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
2020-10-09T00:38:22+08:00
Running ./hw1p1_benchmark
Run on (8 X 3900 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 256 KiB (x4)
  L3 Unified 6144 KiB (x1)
Load Average: 0.79, 2.87, 4.36
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
benchmark_generic_sum                  6949210 ns      6949162 ns          113
benchmark_generic_sum_omp/1          142407562 ns    142407063 ns            5
benchmark_generic_sum_omp/2          269709750 ns    269550336 ns            3
benchmark_generic_sum_omp/3          421977042 ns    421975665 ns            2
benchmark_generic_sum_omp/4          526719105 ns    460071960 ns            2
benchmark_generic_sum_omp/5          593159908 ns    573806184 ns            1
benchmark_generic_sum_omp/6          612195650 ns    479572759 ns            2
benchmark_generic_sum_omp/7          694122609 ns    599149453 ns            1
benchmark_generic_sum_omp/8          772283708 ns    622085024 ns            1
benchmark_generic_sum                  7301945 ns      7301877 ns          111
benchmark_generic_sum_omp_lock/1     158243118 ns    158221035 ns            5
benchmark_generic_sum_omp_lock/2     392907801 ns    392906206 ns            2
benchmark_generic_sum_omp_lock/3     539858913 ns    533585783 ns            1
benchmark_generic_sum_omp_lock/4     916058794 ns    915754188 ns            1
benchmark_generic_sum_omp_lock/5     893991603 ns    893940081 ns            1
benchmark_generic_sum_omp_lock/6     994158309 ns    970886172 ns            1
benchmark_generic_sum_omp_lock/7    1073070201 ns    933486731 ns            1
benchmark_generic_sum_omp_lock/8    1146788576 ns   1129554723 ns            1
benchmark_generic_sum                  7323569 ns      7321185 ns          100
benchmark_generic_sum_omp_red/1        5033907 ns      5032728 ns          100
benchmark_generic_sum_omp_red/2        3339826 ns      3337632 ns          270
benchmark_generic_sum_omp_red/3        2519376 ns      2517624 ns          314
benchmark_generic_sum_omp_red/4        1991852 ns      1990572 ns          394
benchmark_generic_sum_omp_red/5        1968084 ns      1966539 ns          315
benchmark_generic_sum_omp_red/6        1992904 ns      1924621 ns          280
benchmark_generic_sum_omp_red/7        1854291 ns      1806090 ns          519
benchmark_generic_sum_omp_red/8        1631152 ns      1542267 ns          363
benchmark_generic_sum                  7511361 ns      7511109 ns           90
benchmark_generic_sum_omp_red_neq/1   10221398 ns     10219527 ns           71
benchmark_generic_sum_omp_red_neq/2    5362525 ns      5362470 ns          140
benchmark_generic_sum_omp_red_neq/3    5846137 ns      5846099 ns          197
benchmark_generic_sum_omp_red_neq/4    4678313 ns      4678290 ns          205
benchmark_generic_sum_omp_red_neq/5    3682724 ns      3682304 ns          183
benchmark_generic_sum_omp_red_neq/6    2444267 ns      2444238 ns          289
benchmark_generic_sum_omp_red_neq/7    4586148 ns      4250226 ns          366
benchmark_generic_sum_omp_red_neq/8    2098312 ns      2026365 ns          365
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