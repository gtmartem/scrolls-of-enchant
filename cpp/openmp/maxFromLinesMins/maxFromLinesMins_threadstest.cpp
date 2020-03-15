/*

Find max from mins of lines.
5th programm on the way to understanding openMP.
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

int main(int argc, char* argv[]) {
    int dim = atoi(argv[1]);

    // create target Matrix:
    double** A = (double**)malloc(dim*sizeof(double*));
    for (int i = 0; i < dim; i++)
        A[i] = (double*)malloc(dim*sizeof(double));
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) 
            A[i][j] = rand()%100;

    // create buffer for min values of lines:
    double* X = (double*)malloc(dim*sizeof(double));

    double buffer = RAND_MAX;

    for (int threadsNum = 2; threadsNum <= 40; threadsNum += 2){
        // number of threads:
        omp_set_num_threads(threadsNum);
        // start timer:
        double start = omp_get_wtime();
    #pragma omp parallel for firstprivate(buffer)
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++)
                if (A[i][j] < buffer) buffer = A[i][j];
            X[i] = buffer;
        }
        double max = -1.0;
    #pragma omp parallel for reduction(max: max)
        for (int i = 0; i < dim; i++)
            if (X[i] > max) max = X[i];
        double end = omp_get_wtime() - start;
        printf("item : %f , threadsNum : %d , time : %f\n", max, threadsNum, end);
    }

    return 0;

}