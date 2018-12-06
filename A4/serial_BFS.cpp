#include <bits/stdc++.h>
using namespace std;

#define MAX_VAL 100000000

int main(int argc, char *argv[]){
    if(argc<2){
        cout << "Usage: " << argv[0] << " <graph_file_name>\n";
        return 0;
    }
    
    ifstream input;
    input.open(argv[1]);

    int nodes, edges;
    input >> nodes;
    input >> edges;

    int i;
    int *d = (int*)calloc(nodes, sizeof(int));
    d[0]=0;
    for(i=1; i<nodes; i++){
        d[i] = MAX_VAL;
    }
    
    int cum_no[nodes];
    input >> i; //ignore 0
    for(i=0; i<nodes; i++){
        input >> cum_no[i];
    }

    int n=0;
    int temp;
    for(i=0; i<edges*2; i++){
        input >> temp;
        if(d[temp] > d[n]+1)
            d[temp] = d[n]+1;
        if(i+1 == cum_no[n])
            n++;
    }

    int depth = 0;
    for(i=0; i<nodes; i++){
        if(d[i] > depth)
            depth = d[i];
    }

    cout << "Depth of the tree is " << depth << endl;
}