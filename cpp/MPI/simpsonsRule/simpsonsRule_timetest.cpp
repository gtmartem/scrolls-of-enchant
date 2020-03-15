/*

Simpsons rule for integral solving.
3st programm on the way to understanding MPI.
Author: gtmartem (https://github.com/gtmartem)

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

double function(double);
double simpsonsRule_MPI(MPI_Comm, double (*)(double), int, int, double, double, int, double, double*);

int main(int argc, char* argv[]) { 
    double start, end;
    int rank, size;
    
    int N = 1000000000;
    double result, sum, step;
    double a = 0.0;
    double b = 1.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    step = (b-a)/N;
    sum = 0;
    for (int i = (rank + 1); i < N; i += size) {
        if (i % 2 != 0) sum += 4*function(a + step * i);
        else            sum += 2*function(a + step * i);
    }
    sum *= step/3;
    MPI_Reduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) result += (step/3)*(function(a) + function(b));

    //simpsonsRule_MPI(MPI_COMM_WORLD, function, rank, size, a, b, N, step, &result);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime() - start;
        printf("Time : %f , N : %d ,  processes : %d\n", end, N, size);
    }

    MPI_Finalize();
    return 0;  
}

double function(double x) {
    return 4 / (pow(x,2) + 1);
}

double simpsonsRule_MPI(MPI_Comm comm,
                        double (*function)(double),
                        int rank,
                        int size,
                        double a, 
                        double b, 
                        int N,
                        double step,
                        double* result) {

    double sum;
    for (int i = (rank + 1); i < N; i += size) {
        if (i % 2 != 0) sum += 4*function(a + step * i);
        else            sum += 2*function(a + step * i);
    }
    sum *= step/3;

    MPI_Reduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0) {
        *result += (step/3)*(function(a) + function(b));
    }
}