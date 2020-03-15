/*

Simpsons rule for integral solving.
4th programm on the way to understanding openMP.
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
#include <math.h>
#include <stdlib.h>

double function(double);
double simpsonsRule_omp(double (*)(double), double, double, int);

int main(int argc, char* argv[]) {

    int a = 0;
    int b = 1;
    int N = atoi(argv[1]);

    for (int threadsNum = 2; threadsNum <= 40; threadsNum += 2) {

        // number of threads:
        omp_set_num_threads(threadsNum);
        // start timer:
        double start = omp_get_wtime();
        double result = simpsonsRule_omp(function, a, b, N);
        double end = omp_get_wtime() - start;
        printf("N : %d , result : %f , time : %f , threadsNum : %d\n", N, result, end, threadsNum);

    }

    return 0;
}

double function(double x) {
    return 4 / (pow(x,2) + 1);
}

double simpsonsRule_omp(double (*function)(double),
                        double a, 
                        double b, 
                        int N) {

    double step = (b - a)/N;
    double sum = 0.0;

#pragma omp parallel for shared(step) reduction(+:sum)
    for (int i = 1; i < N; i++) {
        if (i % 2 != 0) sum += 4*function(a + step * i);
        else sum += 2*function(a + step * i);
    }
    sum *= step/3;
    sum += (step/3)*(function(a) + function(b));
    return sum;
}