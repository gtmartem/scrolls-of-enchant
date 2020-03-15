/*

Simpsons rule for integral solving.
Single process realization with process ticks counter.
Author: gtmartem (https://github.com/gtmartem)

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double function(double);
double simpsonsRule(double (*)(double), double, double, int);

#pragma intrinsic(__rdtsc);
#define START_TIMER int64_t start = __rdtsc();
#define END_TIMER int64_t end = __rdtsc() - start;

int main(int argc, char* argv[]) {

    int a = 0;
    int b = 1;
    int N = atoi(argv[1]);

    START_TIMER;
    double result = simpsonsRule(function, a, b, N);
    END_TIMER;
    printf("result : %f , time : %lld\n", result, end);

}

double function(double x) {
    return 4 / (pow(x,2) + 1);
}

double simpsonsRule(double (*function)(double),
                        double a, 
                        double b, 
                        int N) {

    double step = (b - a)/N;
    double sum = 0.0;

    for (int i = 1; i < N; i++) {
        if (i % 2 != 0) sum += 4*function(a + step * i);
        else sum += 2*function(a + step * i);
    }
    sum *= step/3;
    sum += (step/3)*(function(a) + function(b));
    return sum;
}