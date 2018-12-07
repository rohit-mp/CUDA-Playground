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

    //initializing depths of all nodes
    int i;
    int *d = (int*)calloc(nodes, sizeof(int));
    d[0]=0;
    for(i=1; i<nodes; i++){
        d[i] = MAX_VAL;
    }
    
    //generating an array of cumulative number of edges from vertices
    int cum_no[nodes];
    input >> i; //ignore 0
    for(i=0; i<nodes; i++){
        input >> cum_no[i];
    }

    //generating the graph in vector form
    vector<int> adj[nodes];
    int n=0;
    for(i=0; i<edges*2; i++){
        int temp;
        input >> temp;
        adj[n].push_back(temp);
        while(i+1==cum_no[n])
            n++;
    }
    for(n=0; n<nodes; n++){
        printf("%d -> ",n);
        for(i=0; i<adj[n].size(); i++){
            printf("%d, ",adj[n][i]);
        }
        printf("\n");
    }

    //performing BFS
    int depth = 0;
    int visited[nodes];
    visited[0]=1;
    for(i=1;i<nodes; i++){
        visited[i]=0;
    }
    queue<int> q;
    q.push(0);
    while(!q.empty()){
        int cur = q.front();
        printf("Popping %d\n",cur);
        q.pop();
        for(i=0; i<adj[cur].size(); i++){
            if( !visited[adj[cur][i]] ){
                d[adj[cur][i]] = d[cur] + 1;
                visited[adj[cur][i]] = 1;
                q.push(adj[cur][i]);
                printf("Pushing %d\n",adj[cur][i]);
                if(depth < d[adj[cur][i]]){
                    depth = d[adj[cur][i]];
                }
            }
        }
    }

    for(i=0; i<nodes; i++){
        printf("Depth of node %d is %d\n",i,d[i]);
    }
    cout << "Depth of the tree is " << depth << endl;
}