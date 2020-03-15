#include "stdio.h"
#include "cuda.h"
#include "curand.h"

#define blk 1024
#define cnt 1024*1024
#define n cnt*blk

#define CUDA_DEBUG
#ifdef CUDA_DEBUG
#define CUDA_CHECK_ERROR(err) \
if (err != cudaSuccess) { \
printf("Cuda error: %s\n", cudaGetErrorString(err)); \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__); \
} 
#endif

__global__ void action2(float * a, float * b, float * c) {
    
    float res = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    res = a[i] * b[i];
    atomicAdd(c, res);
}

int main (int argc, char** argv) {

    float *a, *b, *c;
    float res = 0.0, time = 0.0;
    curandGenerator_t gen;
    cudaEvent_t start, stop;
    
    unsigned long size = sizeof(float) * n;
    size_t heapSize = size * 3;

    CUDA_CHECK_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    CUDA_CHECK_ERROR(cudaMalloc((void **) &a,size));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &b,size));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &c,sizeof(float)));

    cudaMemcpy(c, &res, sizeof(float), cudaMemcpyHostToDevice);

    curandGenerateUniform(gen, a, n);
    curandGenerateUniform(gen, b, n);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    action2<<<cnt,blk>>>(a, b, c);

    cudaMemcpy(&res, c, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("res = %f time = %f\n", res, time);

    curandDestroyGenerator(gen);

}