/*

Find max from mins of lines.
Single process realization with process ticks counter.
Author: gtmartem (https://github.com/gtmartem)

*/

#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

void printMatrix(double**, int);

#pragma intrinsic(__rdtsc);
#define START_TIMER int64_t start = __rdtsc();
#define END_TIMER int64_t end = __rdtsc() - start;

int main(int argc, char* argv[]) {
    int dim = atoi(argv[1]);

    // create target Matrix:
    double** A = (double**)malloc(dim*sizeof(double*));
    for (int i = 0; i < dim; i++)
        A[i] = (double*)malloc(dim*sizeof(double));
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) 
            A[i][j] = rand()%100;
    printf("Target matrix A:\n");
    printMatrix(A, dim);

    // create buffer for min values of lines:
    double* X = (double*)malloc(dim*sizeof(double));

    double buffer = RAND_MAX;

    START_TIMER;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++)
            if (A[i][j] < buffer) buffer = A[i][j];
        X[i] = buffer;
    }
    printf("Vector X (mins of lines):\n");
    for (int i = 0; i < dim; i++)
        printf("%f, ", X[i]);
    printf("\n");
    double max = -1.0;
    for (int i = 0; i < dim; i++) 
        if (X[i] > max) max = X[i];
    END_TIMER;

    printf("ticks : %lld , max from lines mins: %f\n", end, max);

    return 0;
}

void printMatrix(double** A, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++)
            printf("%f, ", A[i][j]);
        printf("\n");
    }
}