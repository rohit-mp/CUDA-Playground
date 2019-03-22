## Histogram of array of numbers using Thrust

Thrust is a Standard Template Library for CUDA that contains a Collection of data parallel primitives (eg. vectors) and implementations (eg. Sort, Scan, saxpy) that can be used in writing high performance CUDA code.

This folder contains the implementation of histogram of array of integers using Thrust.

### How to run

```
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc thrustHistogramSort.cu
$ ./a.out input.raw output.raw
```
