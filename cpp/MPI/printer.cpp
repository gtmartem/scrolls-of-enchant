/*

Process's number printer for 10 processes.
1st programm on the way to understanding MPI.
Author: gtmartem (https://github.com/gtmartem)

*/

/* commands on cluster:
mpicxx -o name name.cpp
sbatch -J name -o name.out -n 1 ompi./name
-J: job name
-o: output file
-n: number of processes */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank;
    int size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("I'm %d from %d processes\n", rank, size);

    MPI_Finalize();
}