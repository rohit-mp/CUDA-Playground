#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <bits/stdc++.h>
#include "wb.h"
using namespace std;

template <class T>
void testSolution(T *h_a, T *h_b, T *h_c, int n, float precision=0.0) {

	int errors = 0;
	for(int i=0; i<n; i++)
		if(abs(h_c[i] - h_a[i] - h_b[i]) > precision) {
			errors++;
			if(errors <= 10)
				printf("Test failed at index : %d\n", i);
		}

	if(errors)
		printf("\n%d Tests failed!\n\n", errors);
	else
		printf("All tests passed !\n\n");
}

int main(int argc, char *argv[]) {

	float *hostInput1 = NULL;
	float *hostInput2 = NULL;
	float *hostOutput = NULL;
	int inputLength;

	wbArg_t arguments = wbArg_read(argc, argv);
	char *output = wbArg_getInputFile(arguments, 0);
	char *input1 = wbArg_getInputFile(arguments, 1);
	char *input2 = wbArg_getInputFile(arguments, 2);

	// Import host input data
	ifstream ifile1(input1);
	ifstream ifile2(input2);

	ifile1 >> inputLength;
	ifile2 >> inputLength;

	printf("\nLength of vector : %d\n", inputLength);

	hostInput1 = new float[inputLength];
	hostInput2 = new float[inputLength];

	for(int i=0; i<inputLength; i++) {
		ifile1 >> hostInput1[i];
		ifile2 >> hostInput2[i];
	}

	// Allocate memory to host output
	hostOutput = new float[inputLength];

	// Declare and allocate thrust device input and output vectors and copy to device
	thrust::host_vector<float> h_in1(hostInput1, hostInput1+inputLength);
	thrust::host_vector<float> h_in2(hostInput2, hostInput2+inputLength);
	thrust::device_vector<float> d_in1=h_in1, d_in2=h_in2, d_out(inputLength);

	// Execute vector addition
    thrust::transform(d_in1.begin(), d_in1.end(), d_in2.begin(), d_out.begin(), thrust::plus<float>());

	// Copy data back to host
	thrust::host_vector<float> h_out = d_out;
	thrust::copy(h_out.begin(), h_out.end(), hostOutput);
	
	testSolution(hostInput1, hostInput2, hostOutput, inputLength, 1e-6);

	delete[] hostInput1, hostInput2, hostOutput;
}
