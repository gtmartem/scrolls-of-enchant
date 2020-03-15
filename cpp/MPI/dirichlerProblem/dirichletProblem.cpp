/*

Dirichlet problem for Lagrange equation.
5th programm on the way to understanding MPI.
Author: Getmanskiy Artem (https://github.com/gtmartem)

*/

/* commands on cluster:
mpicxx -o name name.cpp
sbatch -J name -o name.out -n 1 ompi./name
-J: job name
-o: output file
-n: number of processes */

#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char* argv[]) {

    int mpi_rank, mpi_size, k = 0, i_s, i_e;
    double norma, sqrtnorma;
    double a[(1000/50)+2][1000], b[(1000/50)+3][1000];

    MPI_Status status;

    MPI_Init( &argc, &argv );   
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );

    if (mpi_size != 50) MPI_Abort( MPI_COMM_WORLD, 1);
    
    i_s = 1;
    i_e = 1000/mpi_size;

    if (mpi_rank == 0) i_s++;
    if (mpi_rank == mpi_size - 1) i_e--;

    for (int i=1; i<=1000/mpi_size; i++)    
        for (int j=0; j<1000; j++) 
            a[i][j] = 0;

    for (int j=0; j<1000; j++) {
        a[i_s-1][j] = 1;
        a[i_e+1][j] = 1;
    }

    for (int i=1; i<1000/mpi_size; i++) {
        a[i][0] = 1;
        a[i][1000-1] = 1;
    }

    do {
        if (mpi_rank < mpi_size - 1) MPI_Send( a[1000/mpi_size], 1000, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD );
        if (mpi_rank > 0) MPI_Recv( a[0], 1000, MPI_DOUBLE, mpi_rank-1, 0, MPI_COMM_WORLD, &status );
        if (mpi_rank > 0) MPI_Send( a[1], 1000, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD );
        if (mpi_rank < mpi_size - 1) MPI_Recv( a[1000/mpi_size+1], 1000, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &status );
        k++;
        norma = 0.0;
        for (int i=i_s; i<=i_e; i++)
            for (int j=1; j<1000-1; j++) {
                b[i][j] = (a[i][j+1] + a[i][j-1] + a[i+1][j] + a[i-1][j]) / 4.0;
                norma += (b[i][j] - a[i][j]) * (b[i][j] - a[i][j]);
            }

        for (int i=i_s; i<=i_e; i++)
            for (int j=1; j<1000-1; j++)
                a[i][j] = b[i][j];
                
        MPI_Allreduce( &norma, &sqrtnorma, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
        sqrtnorma = sqrt( sqrtnorma );
        if (k % 100 == 0 && mpi_rank == 0 ) printf("T(20,20) : %f , iter : %d , difference : %e\n", a[19][19], k, sqrtnorma);
    } while (sqrtnorma > 1.0e-3);

MPI_Finalize();

}