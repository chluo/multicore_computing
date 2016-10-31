#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <cuda_runtime.h>

#define MAX_ARRAY_SIZE 1000000

/* 
* Read data from ./inp.txt 
* Store the data in (int * data)
* Return the number of elements read into the array
*/ 
int * read_data(int * size) 
{
    FILE * fptr = fopen("./inp_long.txt", "r"); 
    if (!fptr) {
        printf("!! Error in opening data file \n"); 
        exit(1); 
    }
    int cur_array_size = MAX_ARRAY_SIZE; 
    int * buffer = (int *)malloc(cur_array_size * sizeof(int)); 
    
    int i = 0; 
    while (!feof(fptr)) {
        if (fscanf(fptr, "%d,", &buffer[i]) != 1) {
            break; 
        }
        ++i;         
    }
    
	fclose(fptr); 
    *size = i; 
    return buffer; 
}

/* 
* Round up to the nearest power of 2
*/ 
int round_up_pow2(int val) {
    if (val == 0) return 1; 
    int pow2 = 1; 
    while (pow2 < val) {
        pow2 <<= 1; 
    }
    return pow2; 
}

/*
* Calculate the number of threads per block based on array size 
*/ 
int calc_num_thread(int size) {
    int approx = (int)sqrt((double)size); 
    // find the nearest power of 2 
    return round_up_pow2(approx); 
}

/* 
* GPU kernel for part a: reduction, getting the min value in a sub-array
*/ 
__global__ void shmem_reduce_kernel(int * d_out, const int * d_in, const int size)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ int sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && (myId + s) < size)
        {
            if (sdata[tid] > sdata[tid + s])
                sdata[tid] = sdata[tid + s]; 
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

/* 
* Reduction-based algorithm to find the min value in (int * d_in) 
*/ 
void reduce(int * d_out, int * d_intermediate, int * d_in, int size)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    const int maxThreadsPerBlock = calc_num_thread(size);
    int threads = maxThreadsPerBlock;
    int blocks = (size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>(d_intermediate, d_in, size);

    // now we're down to one block left, so reduce it
    threads = blocks; 
    blocks = 1;
    shmem_reduce_kernel<<<blocks, round_up_pow2(threads), threads * sizeof(int)>>>(d_out, d_intermediate, threads);
}

/* 
* GPU kernel for part b: calculate the last digit of each element in the input array in parallel
*/ 
__global__ void last_digit_kernel(int * d_out, const int * d_in, const int size) 
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    
	if (myId < size)
		d_out[myId] = d_in[myId] % 10; 
}

int main(void)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("!! Error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);
    
    // data array on host 
    int array_size = 0; 
    int * h_in = read_data(&array_size); 
    int array_byte = array_size * sizeof(int);
    // printf(">> Number of data read in: %d\n", array_size); 
    
    /* 
    * part a 
    */ 

    // declare GPU memory pointers
    int * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, array_byte);
    cudaMalloc((void **) &d_intermediate, array_byte); 
    cudaMalloc((void **) &d_out, sizeof(int));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, array_byte, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // launch the kernel
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_in, array_size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // copy back the min from GPU
    int h_out; 
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    // printf(">> Average time elapsed in part a: %f\n", elapsedTime);
    // printf(">> Min value returned by device: %d\n", h_out);
	
	// output the result into file 
    FILE * fptr_a = fopen("./q1a.txt", "w"); 
    if (!fptr_a) {
        printf("!! Error in opening output file \n"); 
        exit(1);
    }
	fprintf(fptr_a, "%d", h_out); 
	fclose(fptr_a); 

    // free GPU memory allocation
    // reuse d_in for the input array of part b
    // reuse d_intermediate for the output array of part b
    cudaFree(d_out);
    
    
    /*
    * part b
    */ 
    
    d_out = d_intermediate; 
    int numThreadPerBlock = calc_num_thread(array_size); 
    int numBlock = (array_size + numThreadPerBlock - 1) / numThreadPerBlock; 
    
    // launch the kernel
    cudaEventRecord(start, 0); 
    last_digit_kernel<<<numBlock, numThreadPerBlock>>>(d_out, d_in, array_size); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf(">> Average time elapsed of part b: %f\n", elapsedTime);
    
    // copy back the result array from GPU
    int * h_out_array = (int *)malloc(array_byte); 
    cudaMemcpy(h_out_array, d_out, array_byte, cudaMemcpyDeviceToHost); 
    
    // output the result array into file 
    FILE * fptr_b = fopen("./q1b.txt", "w"); 
    if (!fptr_b) {
        printf("!! Error in opening output file \n"); 
        exit(1);
    }
    for (int i = 0; i < array_size; ++i) {
        fprintf(fptr_b, "%d", h_out_array[i]); 
        if (i < array_size - 1) 
            fprintf(fptr_b, ", "); 
    }
    fclose(fptr_b); 
    
    // free CPU memory allocation 
    free(h_in); 
    free(h_out_array); 
    
    // free GPU memory allocation 
    cudaFree(d_in); 
    cudaFree(d_intermediate); 

    return 0;
}
