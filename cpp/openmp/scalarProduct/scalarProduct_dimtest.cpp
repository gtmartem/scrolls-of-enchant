/*

Scalar product.
2st programm on the way to understanding openMP.
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
#include <omp.h>
#include <stdlib.h>

void randomizeVector(double*, int);

int main(int argc, char** argv) {
    for (int dim = 1000; dim <= 100000000; dim *= 10) {

        // sum of vectors:
        double sum = 0;

        // vectors:
        double* X = (double*)malloc(dim*sizeof(double));    
        double* Y = (double*)malloc(dim*sizeof(double));
        randomizeVector(X, dim);
        randomizeVector(Y, dim);

        // calculation in threads sections:
        // number of threads:
        // start timer:
        double start = omp_get_wtime();
    #pragma omp parallel for shared(X, Y) reduction(+:sum)
        for (int i = 0; i < dim; i++) sum += X[i] * Y[i];
        double end = omp_get_wtime() - start;
        printf("dimension : %d , time : %f , sum : %f , threads : %d\n", dim, end, sum, omp_get_num_threads());
    }
}

void randomizeVector(double* X, int dim) {
    for (int i = 0; i < dim; i++) X[i] = rand()%100;
}