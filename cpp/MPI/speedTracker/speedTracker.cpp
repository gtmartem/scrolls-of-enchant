/*

Speed tracker.
4st programm on the way to understanding MPI.
Author: gtmartem (https://github.com/gtmartem)

*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h> 

#define MB(L) (L * sizeof(int) * 8)

int main(int argc, char* argv[]) {

    int rank, size;
    double start, end;
    MPI_Status st;
    int L = atoi(argv[1]);

    int transfer[L];
    for (int pos = 0; pos < L; pos++)
                transfer[pos] = rand();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 0; i < 100; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) start = MPI_Wtime();
        if (rank == 0) {
            int transfer_don[L];
            for (int pos = 0; pos < L; pos++) transfer_don[pos] = transfer[pos];
            MPI_Send(&transfer_don, L, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&transfer_don, L, MPI_INT, 1, 0, MPI_COMM_WORLD, &st);
        } else {
            int transfer_acc[L];
            MPI_Recv(&transfer_acc, L, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
            MPI_Send(&transfer_acc, L, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) end = MPI_Wtime() - start;
        if (rank == 0) printf("Time for transfer %d bits = %.10f , so speed : %.10f mb/s\n", MB(L), end, (MB(L)/end/1000000));
    }
    MPI_Finalize();
}
