Q1. RGB image to grayscale image. The input is an RGB triple of float values. You have to convert the triplet to a single float grayscale intensity value. A pseudo-code version of the algorithm is shown
below:  
for ii from 0 to height do  
&nbsp;&nbsp;&nbsp;&nbsp;for jj from 0 to width do  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;idx = ii * width + jj  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;//here channels is 3  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;r = input[3*idx]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;g = input[3*idx + 1]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b = input[3*idx + 2]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// convert  3 r g b values to a single grayscale value.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grayImage[idx] = (0.21*r + 0.71*g + 0.07*b)   
&nbsp;&nbsp;&nbsp;&nbsp;end  
end

Image Format  
The input image is in PPM P6 format while the output grayscale image is to be stored in PPM P5 format. You can create your own input images by exporting your favorite image into a PPM image. On Unix, bmptoppm converts BMP images to PPM images (you could use gimp or similar tools too).
Run command: ```./a.out input.ppm output.pbm```



Q2. Perform Matrix Multiplication of two large integer matrices in CUDA. 

Q3. Implement a tiled matrix multiplication routine using shared memory.