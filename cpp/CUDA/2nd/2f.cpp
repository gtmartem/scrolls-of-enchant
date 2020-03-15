#include "stdio.h"
#include "cuda.h"
#include "curand.h"

#define blk 1024
#define b_p 32
#define s_m 1024
#define cnt 1024*s_m
#define n cnt*blk

#define CUDA_DEBUG
#ifdef CUDA_DEBUG
#define CUDA_CHECK_ERROR(err) \
if (err != cudaSuccess) { \
printf("Cuda error: %s\n", cudaGetErrorString(err)); \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__); \
} 
#endif

__inline__ __device__ float warpReduceSum(float res) {
    
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        res += __shfl_down_sync(0xffffffff,res, offset);
    return res;
}

__inline__ __device__ float blockReduceSum(float res, float *s_v) {
    
    int line = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    res = warpReduceSum(res);
    if (line==0) s_v[wid]=res;
    __syncthreads();
    if(threadIdx.x < blockDim.x / warpSize) res = warpReduceSum(s_v[line]);
    return res;
}

__global__ void action6(float * a, float * b, float * c) {
    
    __shared__ float s_v[b_p];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float res = blockReduceSum(a[i]*b[i],s_v);
    if (threadIdx.x == 0) c[blockIdx.x] = res;
}

__global__ void SumOfVector(float *s_p,float *c) { 
    
    __shared__ float s_v[b_p];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float res = blockReduceSum(s_p[i],s_v);
    if (threadIdx.x == 0) c[blockIdx.x] = res;
}

int main (int argc, char** argv) {

    float *a, *b, *c, *s, *s_p;
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
    CUDA_CHECK_ERROR(cudaMalloc((void **) &s,sizeof(float)*s_m));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &s_p,sizeof(float)*cnt));

    cudaMemcpy(c, &res, sizeof(float), cudaMemcpyHostToDevice);

    curandGenerateUniform(gen, a, n);
    curandGenerateUniform(gen, b, n);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    action6<<<cnt,blk>>>(a, b, s_p);
    SumOfVector<<<cnt / blk, blk>>>(s_p,s);
    SumOfVector<<<1, blk>>>(s,c); 

    cudaMemcpy(&res, c, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("res = %f time = %f\n", res, time);

    curandDestroyGenerator(gen);

}