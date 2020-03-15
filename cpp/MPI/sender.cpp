/*

Process's number sender and printer.
2st programm on the way to understanding MPI.
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
    int rank, size;
    char buf[16];
    MPI_Status st;
    MPI_Request rq;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sender = (rank+9)%10;
    int receiver = (rank+1)%10;
    
    sprintf(buf,"%d",rank);

    MPI_Isend(buf, 16, MPI_CHAR, receiver, 0, MPI_COMM_WORLD, &rq);
    MPI_Recv(buf, 16, MPI_CHAR, sender, 0, MPI_COMM_WORLD, &st);
    
    printf("Process - %d , send to -  %d , recived from - %s , size - %d\n",rank, receiver, buf, size);
    
    MPI_Finalize();

    return 0;
}
