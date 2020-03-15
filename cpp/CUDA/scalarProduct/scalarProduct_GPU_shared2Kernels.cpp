/*

Scalar production.
E 2nd programm on the way to understanding CUDA.
Author: gtmartem (https://github.com/gtmartem)

*/

/* commands on cluster:
nvcc -o NAME NAME.cpp
sbatch -J name -o name.out -n 1 -c 10 run ./name
sbatch -J NAME -o NAME.out --mem=MEMORY_SIZE --gres gpu:COUNT_OF_GPU_UNITS -n 1 run ./NAME
-J: job name
-o: output file
-n: number of runs
--mem=MEMORY_SIZE: memory size
--gres gpu:COUNT_OF_GPU_UNITS: count of GPU units */

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>

#ifndef CUDA_DEBUG
#define CUDA_DEBUG
#define CUDA_CHECK_ERROR (err)                                          \
if (err != cudaSuccess) {                                               \
    printf("Cuda error : %s\n", cudaGetErrorString(err));               \
    printf("Error in file : %s , line : %i\n", __FILE__, __LINE__);     \
}                                                                       
#endif

#define N 10e9

__global__ kernel(float *a, float *b, float *c) {
    __shared__ float cache[minThreadsNumber];
    int threadIndex = treadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    cache[cacheIndex] = a[threadIndex]*b[threadIndex];
    __syncthread();
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex+i];
        __syncthredas();
        i /= 2;
    }
    if (threadIdx.x == 0)
        c[blockIdx.x] = cache[0];
}

__global__ kernel_final(float *c, float *buffer) {
    __shared__ float cache[threadsPerBlock];
    int threadIndex = treadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    cache[cacheIndex] = c[threadIndex];
    __syncthread();
    int i = blockIdx.x/2
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex+i];
        __syncthredas();
        i /= 2;
    }
    if (threadIdx.x == 0)
        buffer[blockIdx.x] = cache[0];
    __syncthredas(); 
}

int main(int argc, char* argv[]) {

    int elapsedTime;
    float *host_a, *host_b, *host_partial_result;
    float host_result;
    float *dev_a, *dev_b, *dev_partial_result;
    float *dev_result;
    curandGenerator_t gen;

    /* Set number of threads */
    CUDA_CALL(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
        CUDA_CALL(cudaGetDeviceProperties(&prop, count));
        if (prop.maxThreadsPerBlock < minThreadsNumber) 
            minThreadsNumber = prop.maxThreadsPerBlock;
    }

    /* Allocate N vectors on host */
    host_a = (float*)malloc(N*sizeof(float));
    host_b = (float*)malloc(N*sizeof(float)); 

    /* Allocate N vectors on device */
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_a, 
                                 N*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_b, 
                                 N*sizeof(float)));
    if (minThreadsNumber == 1024)
        CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_partial_result, 
                                     minThreadsNumber*minThreadsNumber*sizeof(float)));
        host_partial_result = (float*)malloc((int)minThreadsNumber*minThreadsNumber*sizeof(float));
    else
        CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_partial_result, 
                                     (N+minThreadsNumber-1)/minThreadsNumber*sizeof(float))); 
        host_partial_result = (float*)malloc((int)(N+minThreadsNumber-1)/minThreadsNumber*sizeof(float));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_result, 
                                 sizeof(float)));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_PHILOX4_32_10));

    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    /* Generate NxN floats matrix and N floats vector on host */
    CURAND_CALL(curandGenerateUniform(gen, dev_a, N));
    CURAND_CALL(curandGenerateUniform(gen, dev_b, N));

    /* Init timer */
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    /* Start timer */
    CUDA_CALL(cudaEventRecord(start, 0));

    /* Do actions */
    if (minThreadsNumber == 1024)
        kernel<<<minThreadsNumber*minThreadsNumber,minThreadsNumber>>>(dev_a, dev_b, dev_partial_result);
    else
        kernel<<<(N+minThreadsNumber-1)/minThreadsNumber,minThreadsNumber>>>(dev_a, dev_b, dev_partial_result);

    /* Copy results and input on host */
    CUDA_CALL(cudaMempcy(host_a, 
                         dev_a,
                         N*sizeof(float), 
                         cudaMempcyDeviceToHost));
    CUDA_CALL(cudaMempcy(host_b, 
                         dev_b,
                         N*sizeof(float), 
                         cudaMempcyDeviceToHost));
    CUDA_CALL(cudaMempcy(host_partial_result, 
                         dev_partial_result,
                         sizeof(dev_partial_result)/sizeof(dev_partial_result[0])*sizeof(float), 
                         cudaMempcyDeviceToHost));

    int size = sizeof(host_partial_result)/sizeof(host_partial_result[0]);
    blocksPerTask = (size+255/256);
    threadsPerBlock = 256;
    float *dev_buffer;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_buffer, 
                                 blocksPerTask*sizeof(float)));
    while (blocksPerTask > 1) {
        kernel_final<<<blocksPerTask,threadsPerBlock>>>(dev_partial_result, dev_buffer);
        size = sizeof(dev_buffer)/sizeof(dev_buffer[0]);
        blocksPerTask = (size+255/256); 
        CUDA_CALL(cudaFree(dev_partial_result));
        CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_partial_result, 
                                     size*sizeof(float)));
        CUDA_CALL(cudaMempcy(dev_partial_result,
                             dev_buffer,
                             size*sizeof(float),
                             cudaMemcpyDeviceToDevice));
        CUDA_CALL(cudaFree(dev_buffer));
        CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_buffer, 
                                     blocksPerTask*sizeof(float)));
    }
    
    /* Copy results and input on host */
    CUDA_CALL(cudaMempcy(host_result, 
                         dev_buffer[0],
                         sizeof(float), 
                         cudaMempcyDeviceToHost));

    /* Stop timer */
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime,
                                    start, stop));

    /* Log results */
    printf("result : %f\n", host_result);
    printf("time   : %f\n", elapsedTime);
    
    /* Trash time */
    CUDA_CALL(cudaFree(dev_a));
    CUDA_CALL(cudaFree(dev_b));
    CUDA_CALL(cudaFree(dev_partial_result));
    CUDA_CALL(cudaFree(dev_buffer));
    CUDA_CALL(cudaFree(dev_result));
    CURAND_CALL(curandDestroyGenerator(gen));
    delete [] host_a;
    delete [] host_b;
    delete [] host_partial_result;
    
    /* Return control */
    return 0;
}