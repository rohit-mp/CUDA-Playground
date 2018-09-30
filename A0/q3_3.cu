#include<stdio.h>
#include<math.h>

const int MAX_THREADS = 512;

__global__ void array_sum(float *d_in, float *d_sum){
    int ctr = 2;
    *d_sum=0;
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    while(ctr!=MAX_THREADS*2){
        if(idx%ctr==0){
            d_in[idx]+=d_in[idx+ctr/2];
        }
        ctr*=2;
        __syncthreads;
    }
    atomicAdd(d_sum, d_in[blockDim.x*blockIdx.x]);
}
int main(){

    printf("enter a natural number\n");
    int n;
    scanf("%d",&n);
    int array_size = MAX_THREADS*ceil(n/MAX_THREADS);
    int array_bytes = array_size*sizeof(float);

    //generating input array
    float *h_in = (float *)malloc(array_bytes);
    for(int i=0;i<n;i++){
        h_in[i]=i+1;
    }
    h_in[0]=10;
    for(int i=n;i<array_size;i++){
        h_in[i]=0;
    }
    
    //copying data to device
    float *d_in;
    printf("working till here\n");
    cudaMalloc((void **)&d_in, array_bytes);
    cudaMemcpy(d_in, h_in, array_bytes, cudaMemcpyHostToDevice);

    //implementing performance metrics
    float time=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //calling the kernel
    float *d_sum;
    cudaMalloc((void **)&d_sum, sizeof(float));
    cudaEventRecord(start);
    array_sum<<<array_size/MAX_THREADS, MAX_THREADS>>>(d_in, d_sum);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("working\n");
    //copying answer from device to host
    float h_sum[1];
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum of first %d natural numbers is %f\n",n ,h_sum[0]);
    printf("time spent in the gpu : %f\n",time);

    //freeing memory
    cudaFree(d_in);
    cudaFree(d_sum);

    return 0;






}