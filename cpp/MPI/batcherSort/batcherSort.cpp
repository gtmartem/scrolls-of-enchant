/*

Batcher sort algorithm.
7th programm on the way to understanding MPI.
Author: Getmanskiy Artem (https://github.com/gtmartem)

*/

/* commands on cluster:
mpicxx -o name name.cpp
sbatch -J name -o name.out -n 1 ompi./name
-J: job name
-o: output file
-n: number of processes */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void join(int *, int *, int, int, int);
void sortJoin(int *, int *, int, int);

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    int* a = (int*)malloc(n*sizeof(int));
    int mpi_rank, mpi_size, size;
    int* sorted_a; 
    int* add_a;
    double start, end;
    for(int i = 0; i < n; i++) a[i] = rand();
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    if(mpi_rank == 0) {
        printf("Input vector a:\n");
        for(int i = 0; i < n; i++) {
            printf("%d, ", a[i]);
            if (i%9 == 0 && i != 0) printf("\n");
            if (i == (n - 1)) printf("%d\n", a[i]);
        }
    }
    
    size = n/mpi_size;
    int* b = (int*)malloc(size*sizeof(int));
    int* c = (int*)malloc(size*sizeof(int));
    if(mpi_rank == 0) sorted_a =(int*)malloc(n*sizeof(int));

    MPI_Scatter(a, size, MPI_INT, b, size, MPI_INT, 0, MPI_COMM_WORLD);
    sortJoin(b, c, 0, (size - 1));
    MPI_Gather(b, size, MPI_INT, sorted_a, size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(mpi_rank == 0) {
        start = MPI_Wtime();
        add_a = (int*)malloc(n*sizeof(int));
        sortJoin(sorted_a, add_a, 0, (n - 1));
        printf("Sorted vector a:\n");
        for(int i = 0; i < n; i++) {
            printf("%d, ", sorted_a[i]);
            if (i%9 == 0 && i != 0) printf("\n");
            if (i == (n - 1)) printf("%d\n", sorted_a[i]);
        }
        end = MPI_Wtime() - start;
        printf("Time : %f , processes : %d , size(a) : %d\n", end, mpi_size, n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

}

void join(int *a, int *b, int l, int m, int r) {
    int h = l, i = l, j = m+1, k;
    while ((h <= m) && (j <= r)) { 
        if(a[h] <= a[j]) { 
            b[i] = a[h]; h++;
        } else { 
            b[i] = a[j]; j++; 
        }
        i++;
    }
    
    if(m < h) {
        for(k = j; k <= r; k++) { 
            b[i] = a[k]; i++;
        }
    } else {
        for(k = h; k <= m; k++) { 
            b[i] = a[k]; i++; 
        }
    }

    for(k = l; k <= r; k++) a[k] = b[k];
}

void sortJoin(int *a, int *b, int l, int r) {
    int m;
    if (l < r) {
        m = (l+r)/2;
        sortJoin(a, b, l, m);
        sortJoin(a, b, (m+1), r);
        join(a, b, l, m, r);
    }
}