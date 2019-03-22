## Gaussian Blur for images  

An image is represented as RGB float values. The code operates directly on the RGB float values and uses a 3x3 Box Filter to blur the original image to produce the blurred image (Gaussian Blur).  

### How to run

```
g++ dataset_generator.cpp  
./a.out  
nvcc imageBlur.cu -o imageBlur  
./imageBlur input.ppm output.ppm <results.ppm>  
```  

where ```<results.ppm>``` is an optional path to store the results.  
