#include<stdio.h>
#include<stdlib.h>

const int TILE_DIM = 32;
const int MAX_DIM = 100;
const int MAX_SIZE = MAX_DIM*MAX_DIM;
const int MAX_BYTES = MAX_SIZE * sizeof(float);
#define CEIL(a,b) ((a-1)/b + 1)

__global__ void matrix_mul(float *d_m1, float *d_m2, float *d_m3){
    __shared__ float a[TILE_DIM][TILE_DIM], b[TILE_DIM][TILE_DIM];

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int temp=0;

    for(int k=0; k<CEIL(MAX_DIM,TILE_DIM); k++){
        if(x<MAX_DIM && k*TILE_DIM + threadIdx.y<MAX_DIM)
            a[threadIdx.x][threadIdx.y] = d_m1[x*MAX_DIM + k*TILE_DIM + threadIdx.y];
        else
            a[threadIdx.x][threadIdx.y] = 0;
        if(k*TILE_DIM + threadIdx.x<MAX_DIM && y<MAX_DIM)
            b[threadIdx.x][threadIdx.y] = d_m2[k*TILE_DIM*MAX_DIM + threadIdx.x*MAX_DIM + y];
        else
            b[threadIdx.x][threadIdx.y]=0;
        __syncthreads();

        for(int q=0; q<TILE_DIM;q++)
            temp+=a[threadIdx.x][q]*b[q][threadIdx.y];
        __syncthreads();
    }
    if(x<MAX_DIM && y<MAX_DIM)
        d_m3[x*MAX_DIM+y] = temp;
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
    matrix_mul<<< dim3(CEIL(MAX_DIM,32), CEIL(MAX_DIM,32), 1), dim3(32,32,1) >>>(d_m1, d_m2, d_m3);

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
    return 0;
}