/*

Matrix multiplier.
Single process realization with process ticks counter.
Author: gtmartem (https://github.com/gtmartem)

*/


#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

void printMatrix(double**, int);
double** matrixTranspose(double**, int);
double scalarProduct(double*, double*, int);

#pragma intrinsic(__rdtsc);
#define START_TIMER int64_t start = __rdtsc();
#define END_TIMER int64_t end = __rdtsc() - start;

int main(int argc, char** argv) {

    // take dim of Matrix from terminal:
    int dim = atoi(argv[1]);

    // create Matrix A: aij = i + j:
    double** A = (double**)malloc(dim*sizeof(double*));
    for (int i = 0; i < dim; i++)
        A[i] = (double*)malloc(dim*sizeof(double));
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            A[i][j] = i + j;
    printMatrix(A, dim);
    printf("<---- BARRIER ---->\n");

    // create Matrix B: bij = i*j:
    double** B = (double**)malloc(dim*sizeof(double*));
    for (int i = 0; i < dim; i++)
        B[i] = (double*)malloc(dim*sizeof(double));
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            B[i][j] = i*j;
    printMatrix(B, dim);
    printf("<---- BARRIER ---->\n");

    // create Matrix C for result of A*B:
    double** C = (double**)malloc(dim*sizeof(double*));
    for (int i = 0; i < dim; i++)
        C[i] = (double*)malloc(dim*sizeof(double));

    // transpose matrix B:
    double** B_T = matrixTranspose(B, dim);
    printMatrix(B_T, dim);
    printf("<---- BARRIER ---->\n");

    START_TIMER
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            C[i][j] = scalarProduct(A[i], B_T[j], dim);
    printMatrix(C, dim);
    printf("<---- BARRIER ---->\n");
    END_TIMER
    printf("tics: %lld\n", end);

}

void printMatrix(double** A, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++)
            printf("%f, ",A[i][j]);
        printf("\n");
    }
}

double** matrixTranspose(double** A, int dim) {
    double** A_T = (double**)malloc(dim*sizeof(double*));
    for (int i = 0; i < dim; i++)
        A_T[i] = (double*)malloc(dim*sizeof(double));
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            A_T[i][j] = A[j][i];
    return A_T;
}

double scalarProduct(double* X, double* Y, int dim) {
    double sum = 0;
    for (int i = 0; i < dim; i++)
        sum += X[i] * Y[i];
    return sum;
}