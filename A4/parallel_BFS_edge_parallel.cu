#include<bits/stdc++.h>
using namespace std;

#define MAX_VAL 100000000

__global__ void compute(int *d_r, int *d_u, int *d_v, int *depth, int *max_depth, int nodes, int edges){
    int idx = threadIdx.x;
    int i;

    __shared__ int done;
    __shared__ int curr_depth;

    for(i=idx; i<nodes; i+=1024){
        depth[i] = MAX_VAL;
    }        
    if(idx==0){
        depth[0] = 0;
        curr_depth = 0;
        done=0;
    }
    __syncthreads();

    while(!done){
        if(idx == 0){
            done = 1;
        }
        __syncthreads();

        for(i=idx; i<nodes; i+=1024){
            if(depth[i] == curr_depth){
                done = 0;
                for(int j = d_r[i]; j<d_r[i+1]; j++){
                    int to = d_v[j];
                    if(depth[to] > curr_depth + 1){
                        depth[to] = curr_depth + 1;
                    }
                }
            }
        }
        if(idx==0 && done==0){
            curr_depth++;
        }
        __syncthreads();
    }
    if(idx == 0)
        *max_depth = curr_depth-1;
}

int main(int argc, char *argv[]){
    if(argc<2){
        cout << "Usage: " << argv[0] << " <graph_file_name>\n";
        return 0;
    }

    ifstream input;
    input.open(argv[1]);

    int nodes, edges, i, j;
    input >> nodes;
    input >> edges;

    int *h_r = (int*)malloc((nodes+1)*sizeof(int));
    int *h_u = (int*)malloc(edges*2*sizeof(int));
    int *h_v = (int*)malloc(edges*2*sizeof(int));

    for(i=0; i<nodes+1; i++){
        input >> h_r[i];
    }
    for(i=0; i<nodes+1; i++){
        for(j=h_r[i]; j<h_r[i+1]; j++){
            h_u[j] = i;
        }
    }
    for(i=0; i<edges*2; i++){
        input >> h_v[i];
    }

    int *d_r, *d_u, *d_v, *d_depth, *max_depth;
    cudaMalloc((void**)&d_r, (nodes+1)*sizeof(int));
    cudaMalloc((void**)&d_u, edges*2*sizeof(int));
    cudaMalloc((void**)&d_v, edges*2*sizeof(int));
    cudaMalloc((void**)&d_depth, nodes*sizeof(int));
    cudaMalloc((void**)&max_depth, sizeof(int));

    cudaMemcpy(d_r, h_r, (nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u, edges*2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, edges*2*sizeof(int), cudaMemcpyHostToDevice);

    printf("Starting Computation\n");
    compute<<< 1,1024 >>> (d_r, d_u, d_v, d_depth, max_depth, nodes, edges);
    printf("Finished computation\n");
    
    int *result = (int *)malloc(sizeof(int));
    cudaMemcpy(result, max_depth, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Depth : %d\n", result[0]);

    
    int *h_depth = (int*) malloc(nodes*sizeof(int));
	cudaMemcpy(h_depth, d_depth, nodes*sizeof(int), cudaMemcpyDeviceToHost);
	int *h_check_depth = (int*)malloc(nodes*sizeof(int));
	freopen(argv[2], "r", stdin);

	for(int i = 0; i < nodes; i++) {
		cin>>h_check_depth[i];
	}
	bool flag = true;
	int count = 0;

	for(int i = 0; i < nodes; i++) {
		if(h_depth[i] != h_check_depth[i]) {
            printf("Found %d, Expected %d\n",h_depth[i], h_check_depth[i]);
			flag = false;
			count++;
		}
	}

	if(flag) {
		cout<<"Solution is correct!\n";
	}
	else {
		cout<<"Solution is incorrect!"<<endl;
		cout<<count<<" testcases failed.\n";
	}
}