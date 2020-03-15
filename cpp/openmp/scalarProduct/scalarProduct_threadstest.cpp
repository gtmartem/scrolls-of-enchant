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

    // dimension of vectors:
    int dim = atoi(argv[1]);

    // sum of vectors:
    double sum;

    // vectors:
    double* X = (double*)malloc(dim*sizeof(double));    
    double* Y = (double*)malloc(dim*sizeof(double));
    randomizeVector(X, dim);
    randomizeVector(Y, dim);

    int thread;

    // calculation in threads sections:
    for (int threadsNum = 2; threadsNum <= 40; threadsNum += 2) {
        sum = 0;
        // number of threads:
        omp_set_num_threads(threadsNum);
        // start timer:
        double start = omp_get_wtime();
    #pragma omp parallel for shared(X, Y) reduction(+:sum) lastprivate(thread)
        for (int i = 0; i < dim; i++) {
            sum += X[i] * Y[i];
            thread = omp_get_num_threads();
        } 
        double end = omp_get_wtime() - start;
        printf("threads : %d , time : %f , sum : %f , dim : %d\n", thread, end, sum, dim);
    }
}

void randomizeVector(double* X, int dim) {
    for (int i = 0; i < dim; i++) X[i] = rand()%100;
}