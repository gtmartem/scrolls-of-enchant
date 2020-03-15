/*

A * B = C.
6th programm on the way to understanding MPI.
Author: Getmanskiy Artem (https://github.com/gtmartem)

*/

/* commands on cluster:
mpicxx -o name name.cpp
sbatch -J name -o name.out -n 1 ompi./name
-J: job name
-o: output file
-n: number of processes */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int* A; 
    int* B; 
    int* C;
    int* add_B;
    int* add_C; 
    int* sender; 
    int* change; 
    int AN = 100, AM = 15;
    int BN = 15, BM = 7;
    int A_size = AN*AM; 
    int B_size = BN*BM; 
    int C_size = AN*BM;
    int B_change, Br, Bc;
    int size_of_blocks, mpi_size, mpi_rank;
    MPI_Status st;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    size_of_blocks = AM*BM/mpi_size;
    A = (int*)malloc(A_size*sizeof(int));
    add_B = (int*)malloc(B_size*sizeof(int));
    add_C = (int*)malloc(C_size*sizeof(int));
    
    if(mpi_rank == 0) {
        B = (int*)malloc(B_size*sizeof(int));
        C = (int*)malloc(C_size*sizeof(int));
        sender = (int*)malloc(mpi_size*sizeof(int));
        change = (int*)malloc(mpi_size*sizeof(int));
        
        for (int i = 0; i < AN; i++)
            for(int j = 0; j < AM; j++)
                A[i * AM + j] = i + j;
        
        for (int i = 0; i < BN; i++)
            for(int j = 0; j < BM; j++)
                B[i * BM + j] = i * j;
                
        for (int i = 0; i < C_size; i++) {
            add_C[i] = 0;
            C[i] = 0;
        }
        
        for (int i = 0; i < mpi_size - 1; i++) {
            sender[i] = size_of_blocks;
            change[i] = i * size_of_blocks;
        }

        sender[mpi_size - 1] = B_size - (mpi_size - 1) * size_of_blocks;
        change[mpi_size - 1] = (mpi_size - 1) * size_of_blocks;
    }
    
    MPI_Bcast(A, AN * AM, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(add_C, C_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(sender, 1, MPI_INT, &B_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(change, 1, MPI_INT, &B_change, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    add_B = (int*)malloc(B_size*sizeof(int));
    
    MPI_Scatterv(B, sender, change, MPI_INT, add_B, B_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    for(int i = 0; i < B_size; i++) {
        Br = (B_change + i) / BM;
        Bc = (B_change + i) % BM;
        for(int j = 0; j < AN; j++)
            add_C[j * BM + Bc] += A[j * AM + Br] * add_B[i];
    }

    MPI_Reduce(add_C, C, C_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(mpi_rank == 0) {
        printf("A*B product : \n");
        for(int i = 0; i < AN; i++) {
            for (int j = 0; j < BM; j++) 
                printf("%d, ", C[i * BM + j]);
            printf("\n");
        }
    }

    MPI_Finalize();

}