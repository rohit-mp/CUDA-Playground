## Addition of 2 vectors using Thrust

Thrust is a Standard Template Library for CUDA that contains a Collection of data parallel primitives (eg. vectors) and implementations (eg. Sort, Scan, saxpy) that can be used in writing high performance CUDA code.  

This folder contains the implementation of addition of 2 single precision floating point vectors using Thrust.  

### How to run

```
g++ dataset_generator.cpp  
./a.out  
nvcc thrustAdd.cu -o thrustAdd  
./thrustAdd output.raw input0.raw input1.raw <results.raw>  
```  
where ```<results.raw>``` is an option path to store the results.
