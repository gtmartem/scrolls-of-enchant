/*

Scalar production.
B 2nd programm on the way to understanding CUDA.
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

__global__ kernel(float *a, float *b, float *result) {
    float dev_result = 0;
    int threadIndex = treadIdx.x + blockIdx.x * blockDim.x;
    dev_result = a[threadIndex]*b[threadIndex];
    atomicAdd(result, dev_result);
}

int main(int argc, char* argv[]) {

    int elapsedTime;
    float *host_a, *host_b;
    float host_result;
    float *dev_a, *dev_b;
    float *dev_result;
    curandGenerator_t gen;

    /* Allocate N vectors on host */
    host_a = (float*)malloc(N*sizeof(float));
    host_b = (float*)malloc(N*sizeof(float));

    /* Allocate N vectors on device */
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_a, 
                                 N*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_b, 
                                 N*sizeof(float)));
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
    
    /* Set number of threads */
    CUDA_CALL(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
        CUDA_CALL(cudaGetDeviceProperties(&prop, count));
        if (prop.maxThreadsPerBlock < minThreadsNumber) 
            minThreadsNumber = prop.maxThreadsPerBlock;
    }    

    /* Start timer */
    CUDA_CALL(cudaEventRecord(start, 0));

    /* Do actions */
    if (minThreadsNumber == 1024)
        kernel<<<minThreadsNumber*minThreadsNumber,minThreadsNumber>>>(dev_A, dev_b, dev_result);
    else
        kernel<<<(N+minThreadsNumber-1)/minThreadsNumber,minThreadsNumber>>>(dev_A, dev_b, dev_result);

    /* Copy results and input on host */
    CUDA_CALL(cudaMempcy(host_a, 
                         dev_a,
                         N*sizeof(float), 
                         cudaMempcyDeviceToHost));
    CUDA_CALL(cudaMempcy(host_b, 
                         dev_b,
                         N*sizeof(float), 
                         cudaMempcyDeviceToHost));
    CUDA_CALL(cudaMempcy(host_result, 
                         dev_result,
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
    CUDA_CALL(cudaFree(dev_result));
    CURAND_CALL(curandDestroyGenerator(gen));
    delete [] host_a;
    delete [] host_b;
    
    /* Return control */
    return 0;
}