#include "stdio.h"
#include "cuda.h"
#include "curand.h"

#define n 1024

__global__ void action_c(float * a, float * x, float * b) {
    __shared__ float shared_x[n];
    int i = threadIdx.x;
    shared_x[i] = x[i];
    __syncthreads();
    i = threadIdx.x;
    for (int j = 0; j < n; j++) 
        b[i]+= a[i*n+j] * shared_x[j];
}

int main (int argc, char** argv) {

    float *x;
    float *a;
    float *b;
    float time = 0, s = 0;
    curandGenerator_t gen;
    cudaEvent_t start, stop;
    float *res = (float *) malloc(n*sizeof(float));

    cudaMalloc((void **) &a, n*n*sizeof(float));
    cudaMalloc((void **) &b, n*sizeof(float));
    cudaMalloc((void **) &x, n*sizeof(float));

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    curandGenerateUniform(gen, x, n);
    curandGenerateUniform(gen, a, n*n);

    for (int i = 0; i < n; i++) 
        res[i] = 0.;
    
    cudaMemcpy(b, res, n*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    action_c<<<1, n>>>(a, x, b);

    cudaMemcpy(res, b, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    for(int i = 0; i < n; i++)
        s += res[i];

    printf("s = %f time = %f\n", s, time);

    cudaFree(x);
    cudaFree(a);
    cudaFree(b);
    curandDestroyGenerator(gen);

}