/*

Matrix multiplier.
3rd programm on the way to understanding openMP.
Author: gtmartem (https://github.com/gtmartem)

*/

/* commands on cluster:
g++ -fopenmp -o name name.cpp
sbatch -J name -o name.out -n 1 -c 10 run ./name
-J: job name
-o: output file
-n: number of runs
-c: number of threads */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double** matrixTranspose(double**, int);
double scalarProduct(double*, double*, int);

int main(int argc, char* argv[]) {
    for (int dim = 10; dim <= 10000; dim *= 2) {

        // create Matrix A: aij = i + j:
        double** A = (double**)malloc(dim*sizeof(double*));
        for (int i = 0; i < dim; i++)
            A[i] = (double*)malloc(dim*sizeof(double));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                A[i][j] = i + j;

        // create Matrix B: bij = i*j:
        double** B = (double**)malloc(dim*sizeof(double*));
        for (int i = 0; i < dim; i++)
            B[i] = (double*)malloc(dim*sizeof(double));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                B[i][j] = i*j;

        // create Matrix C for result of A*B:
        double** C = (double**)malloc(dim*sizeof(double*));
        for (int i = 0; i < dim; i++)
            C[i] = (double*)malloc(dim*sizeof(double));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                C[i][j] = 0;

        // transpose matrix B:
        double** B_T = matrixTranspose(B, dim);

        // number of threads:
        omp_set_num_threads(10);
        // start timer:
        double start = omp_get_wtime();
    #pragma omp parallel for shared(A, B, C)
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) 
                C[i][j] += scalarProduct(A[i], B_T[j], dim);
        }
        double end = omp_get_wtime() - start;
        printf("dim : %d , time : %f\n", dim, end);
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