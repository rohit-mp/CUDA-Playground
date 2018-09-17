#include<stdio.h>

const int MATRIX_WIDTH = 400;
    const int MATRIX_BYTES = MATRIX_WIDTH * MATRIX_WIDTH * sizeof(float);
    const int MAX_NO_THREADS = 512;

__global__ void matrix_add(float *d_in1, float *d_in2, float *d_out){
    int index = threadIdx.x + blockIdx.x*blockDim.x ;
    *(d_out+index) = *(d_in1+index) + *(d_in2+index);
}

int check(float *h_in1, float *h_in2, float *h_out){
    int flag=1;
    for(int i=0;i<MATRIX_WIDTH*MATRIX_WIDTH;i++){
	if(h_in1[i]+h_in2[i]!=h_out[i])
	    break;
    }
    return flag;
}

int main(){

    //allocating size for host matrices
    float h_in1[MATRIX_WIDTH*MATRIX_WIDTH], h_in2[MATRIX_WIDTH*MATRIX_WIDTH], h_out[MATRIX_WIDTH*MATRIX_WIDTH];

    //generating the input matrices
    int i;
    for(i=0;i<MATRIX_WIDTH*MATRIX_WIDTH;i++){
        h_in1[i]=(float)i;
        h_in2[i]=(float)(MATRIX_WIDTH*MATRIX_WIDTH-i);
    }

    //declaring device memory pointers
    float *d_in1, *d_in2, *d_out;

    //allocating device memory
    cudaMalloc(&d_in1, MATRIX_BYTES);
    cudaMalloc(&d_in2, MATRIX_BYTES);
    cudaMalloc(&d_out, MATRIX_BYTES);

    //transferring memory from host to device
    cudaMemcpy(d_in1, h_in1, MATRIX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, MATRIX_BYTES, cudaMemcpyHostToDevice);

    //starting kernel
    matrix_add<<<(int)(MATRIX_WIDTH*MATRIX_WIDTH/MAX_NO_THREADS)+1, MAX_NO_THREADS>>>(d_in1, d_in2, d_out);

    //transferring memory from device to host
    cudaMemcpy(h_out, d_out, MATRIX_BYTES, cudaMemcpyDeviceToHost);

    //printing the output
    /*for(i=0;i<MATRIX_WIDTH*MATRIX_WIDTH;i++)
        printf("%f\t",(h_out+i))
    printf("\n");*/

    if(check(h_in1,h_in2,h_out))
	printf("the result is correct\n");
    else
	printf("the result is incorrect\n");

    //freeing memory
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);

    return 0;
}
