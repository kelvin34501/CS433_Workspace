#include "device_launch_parameters.h"
#include <iostream>
using namespace std;

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        cout << "GPU device: " << i << ": " << devProp.name << endl;
        cout << "Total Globam Mem: " << devProp.totalGlobalMem / 1024 / 1024
             << "MB" << endl;
        cout << "SM #num: " << devProp.multiProcessorCount << endl;
        cout << "Shared Mem Per Block: " << devProp.sharedMemPerBlock / 1024.0
             << " KB" << endl;
        cout << "Max Threads Per Block: " << devProp.maxThreadsPerBlock << endl;
        cout << "Regs Per Block: " << devProp.regsPerBlock << endl;
        cout << "Max Threads Per SM: " << devProp.maxThreadsPerMultiProcessor
             << endl;
        cout << "Max Warps Per SM: " << devProp.maxThreadsPerMultiProcessor / 32
             << endl;
    }
    cudaDeviceReset();
}