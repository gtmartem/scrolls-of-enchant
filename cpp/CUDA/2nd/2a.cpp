#include "stdio.h"
#include "cuda.h"
#include "curand.h"

#define blk 1024
#define c 1024*1024
#define n c*blk

void action(float* a, float* b, float* res) {
    for(int i = 0; i < n; i++) 
        *res += a[i] * b[i];
}

int main (int argc, char** argv) {
    
    float *a;
    float *b;
    float res = 0.0, time = 0.0;
    curandGenerator_t gen;
    cudaEvent_t start, stop;

    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    curandGenerateUniform(gen, a, n);
    curandGenerateUniform(gen, b, n);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    action(a, b, &res);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("s = %f time = %f\n", res, time);

    curandDestroyGenerator(gen);

}