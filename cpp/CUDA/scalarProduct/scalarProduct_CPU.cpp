/*

Scalar production.
A 2nd programm on the way to understanding CUDA.
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
#include <chrono>

#define N 10e9

int main(int argc, char* argv[]) {

    int elapsedTime;
    float *host_A, *host_b;
    float host_result;
    curandGenerator_t gen;
    std::chrono::time_point<std::chrono::system_clock> start, stop;

    /* Allocate NxN matrix and N vectors on host */
    host_a = (float*)malloc(N*sizeof(float));
    host_b = (float*)malloc(N*sizeof(float));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGeneratorHost(&gen,
                CURAND_RNG_PSEUDO_PHILOX4_32_10));

    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    /* Generate NxN floats matrix and N floats vector on host */
    CURAND_CALL(curandGenerateUniform(gen, host_a, N));
    CURAND_CALL(curandGenerateUniform(gen, host_b, N));

    /* Start timer */
    start = std::chrono::system_clock::now();

    for (int j = 0; j < N; j++)
        host_result += host_a[i] * host_b[i];

    /* End timer */
    stop = std::chrono::system_clock::now();
    elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();

    /* Log results */
    printf("result : %f\n", host_result);
    printf("time   : %f\n", elapsedTime);
    
    /* Trash time */
    CURAND_CALL(curandDestroyGenerator(gen));
    delete [] host_A;
    delete [] host_b;
    
    /* Return control */
    return 0;
}