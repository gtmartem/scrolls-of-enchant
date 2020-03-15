/*

Jacobi iterative method for determining the solutions of a diagonally 
domination system of linear equations.
8th programm on the way to understanding MPI.
Author: Getmanskiy Artem (https://github.com/gtmartem)

*/

#include <iomanip>
#include <math.h>
#include <typeinfo>
#include <cstdlib>
#include <mpi.h>

#define MAXITERS 1000

void init(int, double *, double *, double*);
double evalDiff(double *, double *, int);
void matvec(double *, double *, double *, double *, int, int);

int main(int argc, char* argv[]) {
    int m, np, rk, chunk, i, I = 0, *chunks, *disps;
    double *B, *Bloc, *g, *gloc, *x, *xloc, *xold, t, diff, eps;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);

    if (rk == 0) {
        m = atoi(argv[1]); 
        eps = 0.001; //atof(argv[2]);
        chunk = m/np;
        B = (double*)malloc(m*m*sizeof(double));
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chunk, 1, MPI_INT, 0, MPI_COMM_WORLD);

    chunks = (int*)malloc(np*sizeof(int));
    disps = (int*)malloc(np*sizeof(int));
    g = (double*)malloc(m*sizeof(double));
    x = (double*)malloc(m*sizeof(double));
    xold = (double*)malloc(m*sizeof(double));

    if (rk == 0) {
        init(m, B, g, xold);
        t = MPI_Wtime();
    }
    for (i = 0; i <np; i++) {
        disps[i] = i * chunk * m;
        if (i == (np - 1)) chunks[i] = (m - (np - 1) * chunk)*m;
        else chunks[i] = chunk * m;
    }

    Bloc = (double*)malloc(chunks[rk]*sizeof(double));
    MPI_Scatterv(B,chunks,disps,MPI_DOUBLE,Bloc,chunks[rk],MPI_DOUBLE,0,MPI_COMM_WORLD);

    for (i = 0; i < np; i++) {
        disps[i] = i * chunk;
        if (i == (np - 1)) chunks[i] = m - (np - 1) * chunk;
        else chunks[i] = chunk;
    }

    gloc = (double*)malloc(chunks[rk]*sizeof(double));
    MPI_Scatterv(g,chunks,disps,MPI_DOUBLE,gloc,chunks[rk],MPI_DOUBLE,0,MPI_COMM_WORLD);

    xloc = (double*)malloc(chunks[rk]*sizeof(double));

    do {
        MPI_Bcast(xold,m,MPI_DOUBLE,0,MPI_COMM_WORLD);
        matvec(Bloc, gloc, xold, xloc, chunks[rk], m);
        MPI_Gatherv(xloc, chunks[rk],MPI_DOUBLE,x,chunks,disps,MPI_DOUBLE,0,MPI_COMM_WORLD);
        
        if (rk == 0) {
            diff = evalDiff(xold, x, m);
            //printf("diff = %lf, eps = %lf\n", diff, eps);
            memcpy(xold, x, m * sizeof(double));
        }

        MPI_Bcast(&diff,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        I++;
    } while((diff >= eps) && (I <= MAXITERS));

    if (rk == 0) {
        t = MPI_Wtime() - t;
        printf("Iters : %d , time : %f , size : %d , dim : %d\n", I, t, np, m);
    }

    MPI_Finalize();
}

void init(int m, 
          double *B, 
          double* g, 
          double* x) {
    for (int i = 0; i < m; i++) {
        g[i] = ((double)m+1)/4. + (1.-1./(double)(2*m))*(i+1);
        x[i] = 1.;
        B[i*m + i] = 0.0;
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            B[i*m + j] = 1./(double)(2*m);
}

double evalDiff(double *u, 
                double *v, 
                int m) {
    double a = 0.0;
    for (int i = 0; i < m; i++) {
        double b = v[i] - u[i];
        a += b*b;
    }
    return sqrt(a);
}

void matvec(double *B, 
            double *g, 
            double *xold, 
            double *x, 
            int k, 
            int m) {
    for (int i = 0; i < k; i++) {
        double a = 0.0;
        double *row = B + i*m;
        for (int j = 0; j < m; j++)
            a += row[j] * xold[j];
        x[i] = -a + g[i];
    }
}