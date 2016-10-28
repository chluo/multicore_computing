#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <cuda_runtime.h>

#define MAX_ARRAY_SIZE 1000000

/* -------------------------------------------------------------------------
  Algorithm description: 
  
  array_in      {1, 5, 3, 2, 6, 7, 9, 5, 3, 6} 
     |
	 |  parallel check odd/even: O(1) 
	\_/
  array_is_odd  {1, 1, 1, 0, 0, 1, 1, 1, 1, 0} 
     |
	 |  inclusive prefix scan: O(logN) 
	\_/
  array_index   {1, 2, 3, 3, 3, 4, 5, 6, 7, 7}
  
  num_odd := array_index[N - 1];                       -\
  array_o[num_odd];                                      |
  for i = 1 : N - 1 in parallel                          | O(1)
	if array_is_odd[i]                                   | 
		array_o[array_index[i] - 1] = array_in[i];       |
		                                               -/
  array_o       {1, 5, 3, 7, 9, 5, 3}
  
  T(N) = O(logN) 
------------------------------------------------------------------------- */

/* 
* Check GPU device 
*/ 
void check_dev(void) {
	int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("!! Error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);
}

/*
* Calculate the number of threads per block based on array size 
*/ 
int calc_num_thread(int size) {
	int approx = (int)sqrt((double)size); 
	// find the nearest power of 2 
	int pow2 = 1; 
	while (pow2 < approx) {
		pow2 <<= 1; 
	}
	return pow2; 
}

/* 
* Read data from ./inp.txt 
* Return the pointer to the data array 
* Ouput the number of data items thru passed-in pointer (int * size)
*/ 
int * read_data(int * size) 
{
    FILE * fptr = fopen("./inp.txt", "r"); 
    if (!fptr) {
        printf("!! Error in opening data file \n"); 
        exit(1); 
    }
    int cur_array_size = MAX_ARRAY_SIZE; 
    int * buffer = (int *)malloc(cur_array_size * sizeof(int)); 
    
    int i = 0; 
    while (!feof(fptr)) {
        if (fscanf(fptr, "%d,", &buffer[i]) != 1) {
            printf("!! Error in importing data from file \n"); 
            exit(1); 
        }
        ++i;         
    }
    
	fclose(fptr); 
    *size = i; 
    return buffer; 
}

/* 
* Outputs the result array into file 
*/ 
void print_file(int * array, int array_size) {
	FILE * fptr_b = fopen("./q3.txt", "w"); 
    if (!fptr_b) {
        printf("!! Error in opening output file \n"); 
        exit(1);
    }
    for (int i = 0; i < array_size; ++i) {
        fprintf(fptr_b, "%d", array[i]); 
        if (i < array_size - 1) 
            fprintf(fptr_b, ", "); 
    }
    fclose(fptr_b); 
}

/* 
* GPU kernel: parallel odd/even check
* The output array has 1/odd or 0/even at the corresponding spot
*/ 
__global__ void odd_check(int * array_i, int * array_o, int array_size) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if (myId < array_size) {
		array_o[myId] = array_i[myId] % 2;  
	}
}

/* 
* GPU kernel: inclusive prefix scan 
*/ 
__global__ void prefix_scan(int * array_i, int * array_o, int array_size) {
	// shared memory for intermediate results
	extern __shared__ int sdata[]; 
	
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int thId = threadIdx.x;
	
	// do scan in shared memory 
	int dist = 1; 
	while (dist < array_size) {
		if (!(myId < dist) && myId < array_size) {
			sdata[thId] += array_i[myId - dist]; 
		}
		__syncthreads();  
		dist *= 2; 
	}
	
	// copy the result to the output array 
	if (myId < array_size) {
		array_o[myId] = sdata[thId]; 
	}
}

/* 
* GPU kernel: compact the input array to get the odd numbers 
*/ 
__global__ void get_odd(int * array_i, int * array_o, int * array_is_odd, int * array_index, int array_size, int num_odd) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if (myId < array_size) {
		if (array_is_odd[myId]) {
			array_o[array_index[myId] - 1] = array_i[myId]; 
		}
	}
}

/* 
* Compact algorithm: put the odd numbers in the input array into the output array
* Returns the pointer to the output array
* Ouputs the number of odd numbers thru passed-in pointer (int * num_odd) 
*/ 
int * compact(int * array_i, int * num_odd, int array_size) {
	// dynamically calculate the number of threads and blocks 
	const int maxThreadsPerBlock = calc_num_thread(array_size);
    int threads = maxThreadsPerBlock;
    int blocks = (array_size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
	
	// copy the input array into GPU shared memory 
	int * array_device; 
	cudaMalloc((void **) &array_device, array_size * sizeof(int)); 
	cudaMemcpy(array_device, array_i, array_size * sizeof(int), cudaMemcpyHostToDevice); 
	
	// allocate GPU memories for array_is_odd and array_index 
	int * array_is_odd, * array_index; 
	cudaMalloc((void **) &array_is_odd, array_size * sizeof(int)); 
	cudaMalloc((void **) &array_index, array_size * sizeof(int)); 
	
	// compute array_is_odd 
	odd_check<<<blocks, threads>>>(array_device, array_is_odd, array_size); 
	
	// compute array_index by prefix scan 
	// prefix_scan<<<blocks, threads, threads * sizeof(int)>>>(array_is_odd, array_index, array_size); 
	
	// get the number of odd numbers 
	*num_odd = array_index[array_size - 1]; 
	
	// allocate GPU memory for the result array 
	int * array_device_out; 
	cudaMalloc((void **) &array_device_out, (*num_odd) * sizeof(int)); 
	
	// compute the result
	get_odd<<<blocks, threads>>>(array_device, array_device_out, array_is_odd, array_index, array_size, *num_odd); 
	
	// allocate CPU memory for the result array 
	int * array_o = (int *)malloc((*num_odd) * sizeof(int)); 
	
	// copy the result from GPU to CPU
	cudaMemcpy(array_o, array_device_out, (*num_odd) * sizeof(int), cudaMemcpyDeviceToHost); 
	
	// finish 
	cudaFree(array_device); 
	cudaFree(array_device_out); 
	cudaFree(array_is_odd); 
	cudaFree(array_index); 
	return array_o; 
}

/* 
* CPU main routine 
*/ 
int main(void) {
	// check device 
	check_dev(); 
    
    // data array on host 
    int array_size = 0; 
    int * array_i = read_data(&array_size); 
	
	// do compact 
	int num_odd = 0; 
	int * array_o = compact(array_i, &num_odd, array_size); 
	
	// print to file 
	print_file(array_o, array_size); 
	
	// print debug information to stdout 
	printf(">> Number of odd numbers found: %d\n", num_odd); 
	
	// finish 
	free(array_i); 
	free(array_o); 
	return 0; 
}
