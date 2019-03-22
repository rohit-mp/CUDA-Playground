## Breadth First Search

Breadth-first search is an algorithm for traversing or searching tree or graph data structures. It starts at the tree root, and explores all of the neighbor nodes at the present depth prior to moving on to the nodes at the next depth level.

It has many applications like shortest path, peer to peer networks, crawlers in search engines, etc.

This repository contains the implementation of 3 different approaches to parallelize BFS algorithm. The methods are :
 - Vertex Parallel method
 - Edge Parallel method
 - Work Efficient method

### How to run

```
g++ random_graph_generator.cpp
./a.out <graph_name>
```
Enter the number of nodes and edges when asked
```
g++ serial_BFS.cpp -o serial
./serial <graph_name>
nvcc <program_name.cu> -o <program_name>
./<program_name> <graph_name> output
```

### Performance Comparision

Graph with edges 5 times that of the number of nodes are taken

Number of nodes | Vertex based | Edge based | Work efficient
--- | --- | --- | ---
1e4 | 0.45 | 0.43 | 1.47
1e5 | 5.15 | 4.88 | 17.17
1e6 | 62.08 | 58.54 | 203.53
1e7 | 635.39 | 597.72 | 2071.15
