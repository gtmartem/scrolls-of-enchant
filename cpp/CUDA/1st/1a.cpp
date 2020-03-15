#include "stdio.h"
#include "cuda.h"
#include "curand.h"

int main (int argc, char** argv) {

    float time =0, s = 0;
    int n = 1024;
    
    float *a = (float*) malloc(n*n*sizeof(float));
    float *b = (float*) malloc(n*sizeof(float));
    float *x = (float*) malloc(n*sizeof(float));

    curandGenerator_t gen;
    cudaEvent_t start, stop;

    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    
    curandGenerateUniform(gen, x, n);
    curandGenerateUniform(gen, a, n*n);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) 
            b[i]+= a[i*n+j]*x[j];

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    for(int i = 0; i < n ; i++) 
        s+= b[i];

    cudaEventElapsedTime(&time, start, stop);

    printf("s = %f time = %f\n", s, time);

    curandDestroyGenerator(gen);

}