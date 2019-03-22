#include<stdio.h>

const int ARRAY_SIZE =500000; // size greater than 32M could not be achieved
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
const int MAX_NO_THREADS = 512;

__global__ void vector_reduce(float *d_in1, float *d_in2, float *d_out){
    int index = threadIdx.x + blockIdx.x*blockDim.x ;
    *(d_out+index) = *(d_in1+index) + *(d_in2+index);
}

int check( float *h_in1, float *h_in2, float *h_out){
    int i,flag = 1;
    for(i=0;i<ARRAY_SIZE;i++){
        if(h_in1[i]+h_in2[i]!=h_out[i]){
            flag=0;
            break;
        }
    }
    return flag;
}

int main(){
    

    //allocating size for host arrays
    float h_in1[ARRAY_SIZE], h_in2[ARRAY_SIZE], h_out[ARRAY_SIZE];

    //generating the input arrays
    int i;
    for(i=0;i<ARRAY_SIZE;i++){
        h_in1[i]=(float)i;
        h_in2[i]=(float)(ARRAY_SIZE-i);
    }

    //declaring device memory pointers
    float *d_in1, *d_in2, *d_out;

    //allocating device memory
    cudaMalloc(&d_in1, ARRAY_BYTES);
    cudaMalloc(&d_in2, ARRAY_BYTES);
    cudaMalloc(&d_out, ARRAY_BYTES);

    //transferring memory from host to device
    cudaMemcpy(d_in1, h_in1, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, ARRAY_BYTES, cudaMemcpyHostToDevice);

    //starting kernel
    vector_reduce<<<(int)(ARRAY_SIZE/MAX_NO_THREADS)+1, MAX_NO_THREADS>>>(d_in1, d_in2, d_out);

    //transferring memory from device to host
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    //checking correctness
    if(check(h_in1, h_in2, h_out))
        printf("the result is correct\n");
    else
        printf("the result is incorrect\n");
    
    //freeing memory
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
 
    return 0;
}
