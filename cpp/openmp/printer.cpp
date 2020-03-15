/*

Process's number printer.
1st programm on the way to understanding openMP.
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

int main(int argc, char* argv[]) {
    //int size = atoi(argv[1]);
    #pragma omp parallel //num_threads(size)
        printf("Thread number %d from %d\n", omp_get_thread_num(), omp_get_num_threads());
}

