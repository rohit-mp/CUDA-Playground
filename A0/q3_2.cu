#include<stdio.h>
#include<math.h>

__global__ void add(float *d_in, float *d_out, int array_size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(array_size%2==0 && idx == array_size-1)
        d_out[idx]=d_in[idx];
    else if(idx%2==0)
        d_out[idx/2] = d_in[idx] + d_in[idx+1];

}
int main(){
    
    //reading array size
    printf("Enter the size of array(less than 50k)\n");
    int array_size;
    scanf("%d",&array_size);
    printf("the sum of the first %d natural numbers is ",array_size);

    //allocating memory and generating the array
    float *h_in;
    h_in = (float *)malloc(array_size*sizeof(float));
    for(int i=0; i<array_size; i++){
        h_in[i]=(float)(i+1);
    }

    //allocating memory and copying data to device
    float *d_in, *d_out;
    int array_bytes = array_size*sizeof(float);
    cudaMalloc((void**)&d_in, array_bytes);
    cudaMemcpy(d_in, h_in, array_bytes, cudaMemcpyHostToDevice);

    while(array_size>1){

        cudaMalloc((void **)&d_out, array_bytes/2);
        if(array_size>512)
            add<<<(int)ceil(array_size/512.0f), 512>>>(d_in, d_out, array_size);
        else
            add<<<1, array_size>>>(d_in, d_out, array_size); 
        cudaFree(d_in);
        d_in = d_out;
        array_size = (int)ceil(array_size/2.0f);

    }
    float res[1];
    cudaMemcpy(res, d_in, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n",res[0]);
    return 0;
}