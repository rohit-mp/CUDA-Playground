Q1. Implement vector addition using Thrust. Thrust is a Standard Template Library for CUDA that contains a collection of data parallel primitives (eg. vectors) and implementations (eg. Sort, Scan, saxpy) that can be used in writing high performance CUDA code. Edit the template code to perform the following:
Generate a thrust::dev_ptr<float> for host input arrays
Copy host memory to device
Invoke thrust::transform()
Copy results from device to host

Links for reference:  
http://developer.nvidia.com/technologies/libraries   
http://docs.nvidia.com/cuda/thrust/index.html  
http://www.mariomulansky.de/data/uploads/cuda_thrust.pdf   
https://www.bu.edu/pasi/files/2011/07/Lecture6.pdf   

Instructions about where to place each part of the code is demarcated by the //@@ comment lines. The executable generated as a result of compiling the lab can be run using the following command:
./ThrustVectorAdd_Template <expected.raw> <input0.raw> <input1.raw> <output.raw>
where <expected.raw> is the expected output, <input0.raw>,<input1.raw> is the input dataset, and <output.raw> is an optional path to store the results. The datasets can be generated using the dataset generator. 

Q2. Implement an efficient image blurring algorithm for an input image. An image is represented as `RGB float` values. You will operate directly on the RGB float values and use a 5x5 Box Filter to blur the original image to produce the blurred image (Gaussian Blur). Edit the code in the template to perform the following:
Allocate device memory
Copy host memory to device
Initialize thread block and kernel grid dimensions
Invoke CUDA kernel
Copy results from device to host
Deallocate device memory
The executable generated as a result of compiling the lab can be run using the following command:
./ImageBlur_Template ­ ­<input.ppm> <expected.ppm> <output.ppm>
where <expected.ppm> is the expected output, <input.ppm> is the input dataset, and <output.ppm> is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.

Look up how the Image blur is done in the dataset_generator to essentially parallelize the same in CUDA.
