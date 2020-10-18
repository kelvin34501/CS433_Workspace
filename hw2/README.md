# HW1

## report

see [report](./report.md)

## build instruction

Default build type is ```Release```.
To build:
```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
```
If you don't have ```ninja```, use ```make``` is also possible
```bash
mkdir build && cd build
cmake ..
make -j16
```

To see the results of unit tests: (make sure you are in ```build``` directory)
```bash
./hw2p1_test
./hw2p2_test
```

To see the results of benchmarks: (make sure you are in ```build``` directory)
```bash
./hw2p1_benchmark
./hw2p2_benchmark
```

To change block size and tile size, modify this file:
```bash
nano include/conv_gpu.h
```