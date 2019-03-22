## Converting an RGB image to a gray scale input image

The input image consists of RGB value triples that needs to be converted to a single gray scale image pixel value using the luminosity formula:  
```0.21r + 0.71g + 0.07b```

### How to run  

```
g++ dataset_generator.cpp
./a.out
nvcc imageRGBtoGray.cu
./a.out input.ppm output.ppm
```
