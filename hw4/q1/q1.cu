#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_ARRAY_SIZE 1000000

/* 
* GPU kernel: reduction, getting the min value in a sub-array
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
        if (tid < s && myId < size && (myId + s) < size)
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
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    int blocks = (size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>(d_intermediate, d_in, size);

    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>(d_out, d_intermediate, threads);
}

/* 
* GPU kernel: calculate the last digit of each element in the input array in parallel
*/ 
__global__ void last_digit_kernel(int * d_out, const int * d_in, const int * size) 
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ int sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!
    
	if (myId < size)
		d_out[myId] = sdata[tid] % 10; 
}

/* 
* Read data from ./inp.txt 
* Store the data in (int * data)
* Return the number of elements read into the array
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
    
    *size = i; 
    return buffer; 
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

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }
    
    // data array on host 
    int array_size = 0; 
    int * h_in = read_data(&array_size); 
    int array_byte = array_size * sizeof(int);
    printf(">> Number of data read in: %d\n", array_size); 
    
    /* 
    * Part a 
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

    printf(">> Average time elapsed in part a: %f\n", elapsedTime);
    printf(">> Min value returned by device: %d\n", h_out);

    // free GPU memory allocation
    // Reuse d_in for the input array of part b
    // Reuse d_intermediate for the output array of part b
    cudaFree(d_out);
    
    
    /*
    * Part b
    */ 
    
    d_out = d_intermediate; 
    int numThreadPerBlock = 512; 
    int numBlock = (array_size + numThreadPerBlock - 1) / numThreadPerBlock; 
    
    // launch the kernel
    cudaEventRecord(start, 0); 
    last_digit_kernel<<<numBlock, numThreadPerBlock, numThreadPerBlock * sizeof(int)>>>(d_out, d_in, array_size); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf(">> Average time elapsed of part b: %f\n", elapsedTime);
    
    // copy back the result array from GPU
    int * h_out_array = (int *)malloc(array_size * sizeof(int)); 
    cudaMemcpy(&h_out_array, d_out, array_size * sizeof(int), cudaMemcpyDeviceToHost); 
    
    // output the result array into file 
    FILE * fptr = fopen("./out.txt", "w"); 
    if (!fptr) {
        printf("!! Error in opening output file \n"); 
        exit(1);
    }
    for (int i = 0; i < array_size; ++i) {
        fprintf(fptr, "%d", h_out_array[i]); 
        if (i < array_size - 1) 
            fprintf(fptr, ", "); 
    }
    fclose(fptr); 
    
    // Free CPU memory allocation 
    free(h_in); 
    free(h_out_array); 
    
    // Free GPU memory allocation 
    cudaFree(d_in); 
    cudaFree(d_intermediate); 

    return 0;
}
