#include<stdio.h>
#include<stdlib.h>

#define CEIL(a, b) ((a-1)/b +1)

const int MAX_DIM = 100;
const int MAX_SIZE = MAX_DIM*MAX_DIM;
const int MAX_BYTES = MAX_SIZE*sizeof(float);

__global__ void matrix_mul(float *d_m1, float *d_m2, float *d_m3){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= MAX_SIZE)
        return;
    float tempsum=0;
    int i = idx/MAX_DIM;
    int j = idx%MAX_DIM;
    for(int k=0;k<MAX_DIM;k++){
        tempsum += d_m1[i*MAX_DIM + k]*d_m2[j + k*MAX_DIM];
    }
    d_m3[idx] = tempsum;
}   

int main(){
    //allocating memory for host arrays
    float h_m1[MAX_SIZE], h_m2[MAX_SIZE], h_m3[MAX_SIZE];

    //generating input arrays
    for(int i=0;i<MAX_SIZE;i++)
        h_m1[i] = (float)(rand()%100);
    for(int i=0;i<MAX_SIZE;i++)
        h_m2[i] = (float)(rand()%100);

        //declaring device memory pointers
    float *d_m1, *d_m2, *d_m3;

    //allocating device memory
    cudaMalloc((void **)&d_m1, MAX_BYTES);
    cudaMalloc((void **)&d_m2, MAX_BYTES);
    cudaMalloc((void **)&d_m3, MAX_BYTES);

    //copying data from host to device
    cudaMemcpy(d_m1, h_m1, MAX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, h_m2, MAX_BYTES, cudaMemcpyHostToDevice);

    //calling kernel
    matrix_mul<<< CEIL(MAX_SIZE, 1024), 1024 >>>(d_m1, d_m2, d_m3);

    //transferring result from device to host
    cudaMemcpy(h_m3, d_m3, MAX_BYTES, cudaMemcpyDeviceToHost);

    //checking correctness of answer
    int flag =1;
    for(int i=0;i<MAX_DIM;i++){
        for(int j=0;j<MAX_DIM;j++){
            float tempsum=0;
            for(int k=0;k<MAX_DIM;k++){
                tempsum += h_m1[i*MAX_DIM + k] * h_m2[j + k*MAX_DIM];
            }
            if(h_m3[i*MAX_DIM+j] != tempsum){
                printf("wrong value at %d\n",i*MAX_DIM+j);
                printf("Expected value:%f, found value:%f\n",tempsum, h_m3[i*MAX_DIM+j]);
                flag=0;
                break;
            }
        }
        if(flag==0)
            break;
    }
    if(flag==1)
        printf("The solution is correct\n");
}