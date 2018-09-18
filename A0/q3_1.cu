#include<stdio.h>

__global__ void add(float *d_in, float *d_out){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    d_out[idx] = d_in[idx*2]+d_in[idx*2+1];
}

int main(){
    
    //reading array size
    printf("Enter the size of array(less than 50k)\n");
    int array_size;
    scanf("%d",&array_size);
    printf("the sum of the first %d natural numbers is ",array_size-1);

    //allocating memory and generating the array
    float *h_in;
    h_in = (float *)malloc(array_size*sizeof(float));
    for(int i=0; i<array_size; i++){
        h_in[i]=i;
    }

    //adding zero padding at the end
    int closest_power =(int)pow(2,ceil(log(array_size)/log(2)));
    h_in = (float *)realloc(h_in, closest_power*sizeof(float));
    for(;array_size<closest_power;array_size++)
        h_in[array_size]=0;

    //allocating and copying info to device
    float *d_in ;
    int array_bytes = array_size*sizeof(float);
    cudaMalloc((void **)&d_in, array_bytes);
    cudaMemcpy(d_in, h_in, array_size*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //iteratively calling the kernel and calculating the time
    float time=0;
    while(array_size>1){
        float *d_out;
        float temptime=0;
        array_bytes = array_size*sizeof(float);
        cudaMalloc((void **)&d_out, array_bytes/2);
        cudaEventRecord(start);
        if(array_size>1024)
            add<<<array_size/512+1, 512>>>(d_in, d_out);
        else
            add<<<array_size/512+1, array_size/2>>>(d_in, d_out);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temptime, start, stop);
        time+=temptime;
        cudaFree(d_in);
        d_in = d_out;
        array_size/=2;
    }
    float value[1];
    cudaMemcpy(value,d_in, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("%f\n",*value);
    printf("time taken : %f\n",time);

    return 0;
}   