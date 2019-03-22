## Image Convolution

The code implements an image convolution using a 5 x 5 mask with the tiled shared memory approach. Convolution is used in many fields, such as image processing for image filtering.

### How to run

```
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc imageConvolution.cu
$ ./a.out input.ppm mask.raw output.ppm
```
