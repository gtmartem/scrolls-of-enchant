/*

Scalar product.
Single process realization with process ticks counter.
Author: gtmartem (https://github.com/gtmartem)

*/

#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

void randomizeVector(double*, int);

#pragma intrinsic(__rdtsc);
#define START_TIMER int64_t start = __rdtsc();
#define END_TIMER int64_t end = __rdtsc() - start;

int main(int argc, char** argv) {

    // dimension of vectors:
    int dim = atoi(argv[1]);

    // sum of vectors:
    double sum = 0;

    // vectors:
    double* X = (double*)malloc(dim*sizeof(double));    
    double* Y = (double*)malloc(dim*sizeof(double));
    randomizeVector(X, dim);
    printf("Vector X:\n");
    for (int i = 0; i < dim; i++)
        printf("%f, ", X[i]);
    printf("\n");
    randomizeVector(Y, dim);
    printf("Vector Y:\n");
    for (int i = 0; i < dim; i++)
        printf("%f, ", Y[i]);
    printf("\n");

    START_TIMER;
    for (int i = 0; i < dim; i++) sum += X[i] * Y[i];
    END_TIMER;
    printf("time : %lld , sum : %f\n", end, sum);

    return 0;
}

void randomizeVector(double* X, int dim) {
    for (int i = 0; i < dim; i++) X[i] = rand()%100;
}