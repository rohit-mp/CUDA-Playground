#include "wb.h"
#include <bits/stdc++.h>
using namespace std;

#define wbCheck(stmt)                                               \
	do {                                                            \
		cudaError_t err = stmt;                                     \
		if (err != cudaSuccess) {                                   \
			wbLog(ERROR, "Failed to run stmt ", #stmt);             \
			return -1;                                              \
		}                                                           \
	} while (0);

#define Mask_width 5
#define Mask_radius (Mask_width / 2)
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))
#define CEIL(a, b) ((a-1)/b +1)

const int num_channels = 3;

__global__ void convolve(float *deviceInputImageData, float* __restrict__ deviceMaskData, float *deviceOutputImageData, int width, int height){

	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if(idx>=0 && idx<width*height){
		float color[3]={0,0,0};
		for(int r=-Mask_radius; r<=Mask_radius; r++){
			for(int c=-Mask_radius; c<=Mask_radius; c++){
				for(int channel=0; channel<num_channels; channel++){
					if( c>=-(idx%width) && r>=-(idx/width) && c<width-idx%width && r<height-idx/width)
					//idx + r*num_channels*width[0] + c*num_channels >= 0 && idx + r*num_channels*width[0] + c*num_channels < height[0]*width[0]*num_channels)
						color[channel]+= deviceInputImageData[idx*num_channels + r*num_channels*width + c*num_channels + channel] * deviceMaskData[(r+Mask_radius)*Mask_width + c+Mask_radius];
				}
			}
		}
		deviceOutputImageData[num_channels*idx+0] = clamp(color[0]);
		deviceOutputImageData[num_channels*idx+1] = clamp(color[1]);
		deviceOutputImageData[num_channels*idx+2] = clamp(color[2]);
	}

}

int main(int argc, char *argv[]) {

	wbArg_t arg;
	int maskRows = Mask_width;
	int maskColumns = Mask_width;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile  = wbArg_getInputFile(arg, 1);

	inputImage   = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData  = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, 
		imageHeight * imageWidth * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, 
		imageHeight * imageWidth * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceMaskData, 
		Mask_width * Mask_width * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData, 
		imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutputImageData, hostOutputImageData,
		imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, 
		Mask_width * Mask_width * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	convolve<<< CEIL(imageHeight*imageWidth, 1024), 1024 >>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageWidth, imageHeight);
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//@@ Insert code here
	free(hostInputImageData);
	free(hostOutputImageData);
	free(hostMaskData);

	cudaFree(deviceMaskData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceInputImageData);
}
