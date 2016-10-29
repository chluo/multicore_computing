#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <cuda_runtime.h>

#define MAX_ARRAY_SIZE 1000000

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
void print_file(int * array, int array_size, const char fname[]) {
    FILE * fptr_b = fopen(fname, "w"); 
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
* GPU kernel: inclusive prefix scan, one step 
*/ 
__global__ void prefix_scan_step(int * array_io, int array_size, int dist) {
    // shared memory to store intermediate results 
    extern __shared__ int sdata[]; 
    
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int thId = threadIdx.x; 
    
    // load initial values to shared memory 
    sdata[thId] = array_io[myId]; 
    __syncthreads(); 
    
    // store block results in shared memory 
    if (!(myId < dist) && myId < array_size) {
        sdata[thId] += array_io[myId - dist]; 
    }
    __syncthreads();  
    // copy results to global memory 
    if (myId < array_size) {
        array_io[myId] = sdata[thId]; 
    }
    __syncthreads(); 

}

/* 
* Inclusive prefix scan
*/ 
void prefix_scan(int * array_io, int array_size) {
    // dynamically calculate the number of threads and blocks 
    const int maxThreadsPerBlock = calc_num_thread(array_size);
    int threads = maxThreadsPerBlock;
    int blocks = (array_size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    
    int dist = 1; 
    while (dist < array_size) {
        prefix_scan_step<<<blocks, threads, threads * sizeof(int)>>>(array_io, array_size, dist); 
        cudaThreadSynchronize(); 
        dist *= 2; 
    }
}

/* 
* GPU kernel: reduction, getting the sum of an array 
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
    for (unsigned int s = (blockDim.x + 1) / 2; s > 0; s = (s == 1) ? 0 : (s + 1) / 2)
    {
        if (tid < s && (myId + s) < size)
        {
            if (tid + s < blockDim.x) {
                sdata[tid] += sdata[tid + s]; 
            }
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
* Reduction-based algorithm to find the sum of an array  
*/ 
void reduce(int * d_out, int * d_intermediate, int * d_in, int size)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    const int maxThreadsPerBlock = calc_num_thread(size);
    int threads = maxThreadsPerBlock;
    int blocks = (size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>(d_intermediate, d_in, size); 
    
    // debug 
    int * debug = (int *)malloc(blocks * sizeof(int)); 
    cudaMemcpy(debug, d_intermediate, blocks * sizeof(int), cudaMemcpyDeviceToHost); 
    print_file(debug, blocks, "./debug.txt"); 

    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>(d_out, d_intermediate, threads);
}

/* 
* GPU kernel for part a
*/ 
__global__ void global_counter_kernel(int * array_i, int * cnt_o, int array_size) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x; 
    if (myId < array_size) {
        atomicAdd(&cnt_o[array_i[myId] / 100], 1); 
    }
}

/* 
* part a: global memory counter 
* returns the pointer to the result array B 
*/ 
int * global_counter(int * array_i, int array_size) {
    // dynamically calculate the number of threads and blocks 
    const int maxThreadsPerBlock = calc_num_thread(array_size);
    int threads = maxThreadsPerBlock;
    int blocks = (array_size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    
    // allocate GPU global memories for input & output arrays 
    int * array_device, * array_device_out; 
    cudaMalloc((void **) &array_device, array_size * sizeof(int)); 
    cudaMalloc((void **) &array_device_out, 11 * sizeof(int)); 
    
    // copy the input array into GPU shared memory 
    cudaMemcpy(array_device, array_i, array_size * sizeof(int), cudaMemcpyHostToDevice); 
    
    // launch the kernel 
    global_counter_kernel<<<blocks, threads>>>(array_device, array_device_out, array_size); 
    
    // allocate CPU memory for output array 
    int * array_o = (int *)malloc(10 * sizeof(int)); 
    
    // copy result back to CPU 
    cudaMemcpy(array_o, array_device_out, 10 * sizeof(int), cudaMemcpyDeviceToHost); 
    
    // finish 
    cudaFree(array_device); 
    cudaFree(array_device_out); 
    return array_o; 
}

/* 
* GPU kernel for part b 
* cnt_matrix dimensions: 10 x (# of blocks) 
*/ 
__global__ void shmem_counter_kernel(int * array_i, int * cnt_matrix, int array_size, int num_block) {
    // shared counter within block
    // size: 11 * sizeof(int) 
    // one extra int for numbers greater than 1000 
    extern __shared__ int scnt[]; 
    
    // initialize to 0 
    if (threadIdx.x < 10) {
        scnt[threadIdx.x] = 0; 
    }
    __syncthreads();
    
    // block-local counter 
    int myId = threadIdx.x + blockDim.x * blockIdx.x; 
    if (myId < array_size) {
        atomicAdd(&scnt[array_i[myId] / 100], 1); 
    }
    __syncthreads(); 
    
    // copy the counter values to shared memory 
    // only have 10 values 
    if (threadIdx.x < 10) {
        cnt_matrix[threadIdx.x * num_block + blockIdx.x] = scnt[threadIdx.x]; 
    }
}

/* 
* part b: shared memory counter 
* returns the pointer to the result array B 
*/ 
int * shmem_counter(int * array_i, int array_size) {
    // dynamically calculate the number of threads and blocks 
    const int maxThreadsPerBlock = calc_num_thread(array_size);
    int threads = maxThreadsPerBlock;
    int blocks = (array_size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    
    // allocate GPU global memories for input & output arrays and intermediate counter matrix  
    int * array_device, * array_device_inter, * array_device_out; 
    cudaMalloc((void **) &array_device, array_size * sizeof(int)); 
    cudaMalloc((void **) &array_device_inter, 10 * blocks * sizeof(int)); 
    cudaMalloc((void **) &array_device_out, 10 * sizeof(int)); 
    
    /* --------------------------------------------------------------------------------
     * The intermediate counter matrix 
     *
     *             block 0 | block 1 | block 2 | ... ... | block N* 
     * [  0,  99]
     * [100, 199]
     * [200, 299]
     * ... ... 
     * [900, 999]
     *
     * *Note: the number of blocks, N, is stored in variable "blocks"
    -------------------------------------------------------------------------------- */
    
    // allocate GPU global memory for reduction's intermediate results 
    int * array_device_reduction_inter; 
    cudaMalloc((void **) &array_device_reduction_inter, blocks * sizeof(int)); 
    
    // allocate CPU memory for the output array  
    int * array_o = (int *)malloc(10 * sizeof(int));  
    
    // copy the input array into GPU shared memory 
    cudaMemcpy(array_device, array_i, array_size * sizeof(int), cudaMemcpyHostToDevice); 
    
    // launch the counter kernel 
    // shared memory size: 11 * sizeof(int) 
    // one extra int for numbers greater than 1000 
    shmem_counter_kernel<<<blocks, threads, 11 * sizeof(int)>>>(array_device, array_device_inter, array_size, blocks); 
    
    // do reduction for each range 
    for (int i = 0; i < 10; ++i) {
        reduce(&array_device_out[i], array_device_reduction_inter, &array_device_inter[blocks * i], blocks);  
    }
       
    // copy result back to CPU 
    cudaMemcpy(array_o, array_device_out, 10 * sizeof(int), cudaMemcpyDeviceToHost); 
    
    // finish 
    cudaFree(array_device); 
    cudaFree(array_device_out); 
    cudaFree(array_device_inter); 
    cudaFree(array_device_reduction_inter); 
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
    
    // part a ------------------------------------------------------------ 
    // compute counter values 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int * array_o_a = global_counter(array_i, array_size); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // print to file 
    print_file(array_o_a, 10, "./q2a.txt"); 
    
    // print debug information to stdout 
    printf(">> Average time elapsed in part a: %f\n", elapsedTime);

    
    // part b ------------------------------------------------------------ 
    cudaEvent_t start_b, stop_b;
    cudaEventCreate(&start_b);
    cudaEventCreate(&stop_b);
    cudaEventRecord(start_b, 0);
    int * array_o_b = shmem_counter(array_i, array_size); 
    cudaEventRecord(stop_b, 0);
    cudaEventSynchronize(stop_b);
    cudaEventElapsedTime(&elapsedTime, start_b, stop_b);
        
    // print to file 
    print_file(array_o_b, 10, "./q2b.txt"); 
    
    // print debug information to stdout 
    printf(">> Average time elapsed in part b: %f\n", elapsedTime);
    
    // part c ------------------------------------------------------------ 
    
    // finish 
    free(array_i); 
    free(array_o_a); 
    free(array_o_b); 
    return 0; 
}