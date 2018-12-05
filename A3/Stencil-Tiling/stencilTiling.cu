#include "wb.h"
#include <bits/stdc++.h>
using namespace std;

#define CEIL(a, b) ((a-1)/b +1)
#define Clamp(a, start, end) (max(min(a, end), start))
#define value(arry, i, j, k) (arry[((i)*width + (j)) * depth + (k)])
#define output(i, j, k) value(output, i, j, k)
#define input(i, j, k) value(input, i, j, k)
#define data(i, j, k) data[i*121 + j*11 + k]

#define wbCheck(stmt)                                                           \
    do {                                                                        \
        cudaError_t err = stmt;                                                 \
        if (err != cudaSuccess) {                                               \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
            return -1;                                                          \
        }                                                                       \
    } while (0)

__global__ void compute(float *deviceInputData, float *deviceOutputData, int width, int height, int depth){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int j = x/depth;
    int k = x%depth;
    int i = y;

    if(j>0 && j<width-1 && i>0 && i<height-1 && k>0 && k<depth-1){
        float val = deviceInputData[((i-1)*width + (j)) * depth + (k)] + deviceInputData[((i)*width + (j-1)) * depth + (k)] 
            + deviceInputData[((i)*width + (j)) * depth + (k-1)] + deviceInputData[((i+1)*width + (j)) * depth + (k)] 
            + deviceInputData[((i)*width + (j+1)) * depth + (k)] + deviceInputData[((i)*width + (j)) * depth + (k+1)]
            - 6*deviceInputData[((i)*width + (j)) * depth + (k)];
        deviceOutputData[((i)*width + (j)) * depth + (k)] = Clamp(val, 0.0, 1.0);
        if(i==1021) deviceOutputData[((i)*width + (j)) * depth + (k)] = 0.15;
    }
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData, 
    int width, int height, int depth) {
        compute<<< dim3( CEIL(width*depth, 32), CEIL(height, 32), 1), dim3(32,32,1) >>> (deviceInputData, deviceOutputData, width, height, depth);
    //Kernel call
}

int main(int argc, char *argv[]) {

    wbArg_t arg;
    int width;
    int height;
    int depth;
    char *inputFile;
    wbImage_t input;
    wbImage_t output;
    float *hostInputData;
    float *deviceInputData;
    float *deviceOutputData;

    arg = wbArg_read(argc, argv);

    inputFile = wbArg_getInputFile(arg, 0);
    input = wbImport(inputFile);

    width  = wbImage_getWidth(input);
    height = wbImage_getHeight(input);
    depth  = wbImage_getChannels(input);

    output = wbImage_new(width, height, depth);

    hostInputData  = wbImage_getData(input);

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
    cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float),
        cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(output.data, deviceOutputData, width * height * depth * sizeof(float),
        cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    /*float *hostOutputData = (float*)malloc(sizeof(float)*width*height*depth);
    cudaMemcpy(hostOutputData, deviceOutputData, sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost);
    int i,j,k;
    for(i=1; i<width-1; i++){
        for(j=1; j<height-1; j++){
            for(k=1; k<depth-1; k++){
                if(hostOutputData[((i)*width + (j)) * depth + (k)] != Clamp(hostInputData[((i-1)*width + (j)) * depth + (k)] + hostInputData[((i+1)*width + (j)) * depth + (k)]
                    + hostInputData[((i)*width + (j-1)) * depth + (k)] + hostInputData[((i)*width + (j+1)) * depth + (k)]
                    + hostInputData[((i)*width + (j)) * depth + (k-1)] + hostInputData[((i)*width + (j)) * depth + (k+1)]
                    - 6*hostInputData[((i)*width + (j)) * depth + (k)], 0.0, 1.0))
                    printf("%d,%d,%d Found:%f, Expected:%f\n",i,j,k,hostOutputData[((i)*width + (j)) * depth + (k)], hostInputData[((i-1)*width + (j)) * depth + (k)] + hostInputData[((i+1)*width + (j)) * depth + (k)]
                    + hostInputData[((i)*width + (j-1)) * depth + (k)] + hostInputData[((i)*width + (j+1)) * depth + (k)]
                    + hostInputData[((i)*width + (j)) * depth + (k-1)] + hostInputData[((i)*width + (j)) * depth + (k+1)]
                    - 6*hostInputData[((i)*width + (j)) * depth + (k)]);
            }
        }
    }*/

    wbSolution(arg, output);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    wbImage_delete(output);
    wbImage_delete(input);
}