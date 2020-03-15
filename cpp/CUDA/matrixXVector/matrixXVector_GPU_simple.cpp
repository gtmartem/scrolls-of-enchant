/*

Matrix and vector multiplier.
B 1st programm on the way to understanding CUDA.
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
#include <assert.h>

#define N 1024

__global__ void kernel(float* A, float* b, float *c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = 0; j < N; j++)
        c[i] += A[i*N + j] * b[j];
}

int main(int argc, char* argv[]) {

    float elapsedTime;
    float *dev_A, *dev_b, *dev_c;
    float *host_A, *host_b, *host_c;
    curandGenerator_t gen;
    curandEvent_t start, stop;
    cudaDeviceProp prop;
    int count, minThreadsNumber = 1024;
    float check = 0;

    /* Allocate NxN matrix and N vectors on device */
    CUDA_CALL(cudaMalloc((void**)&dev_A, 
                          N*N*sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_b, 
                          N*sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_c, 
                          N*sizeof(float)));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_PHILOX4_32_10));

    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    /* Generate NxN floats matrix and N floats vector on device */
    CURAND_CALL(curandGenerateUniform(gen, dev_A, N*N));
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
        kernel<<<1,minThreadsNumber>>>(dev_A, dev_b, dev_c);
    else
        kernel<<<(N+minThreadsNumber-1)/minThreadsNumber,minThreadsNumber>>>(dev_A, dev_b, dev_c);

    /* Copy results and input on host */
    CUDA_CALL(cudaMempcy(host_A, 
                         dev_A,
                         N*N*sizeof(float), 
                         cudaMempcyDeviceToHost));
    CUDA_CALL(cudaMempcy(host_b, 
                         dev_b,
                         N*sizeof(float), 
                         cudaMempcyDeviceToHost));
    CUDA_CALL(cudaMempcy(host_c, 
                         dev_c,
                         N*sizeof(float), 
                         cudaMempcyDeviceToHost));

    /* Stop timer */
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime,
                                    start, stop));

    /* Check reult */
    for (int i = 0; i < N; i++)
        check += host_A[5*N + i] * host_b[i];
    assert(host_c[5] == check);

    /* Log time */
    printf("when everything is in global memory:\n")
    printf("time :  %f\n", elapsedTime);

    /* Trash time */
    CUDA_CALL(cudaFree(dev_A));
    CUDA_CALL(cudaFree(dev_b));
    CUDA_CALL(cudaFree(dev_c));
    CURAND_CALL(curandDestroyGenerator(gen));
    delete [] host_A;
    delete [] host_b;
    delete [] host_c;

    /* Return control */
    return 0;
}