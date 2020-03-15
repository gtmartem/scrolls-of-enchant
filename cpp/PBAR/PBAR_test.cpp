#include <unistd.h>
#include <math.h>
#include "PBAR.h"

/*

Testing:

*/

int main(int argc, char** argv) {
    PBAR *pbar = new PBAR(100);
    for (int i = 1; i <= 100; i++) 
    {
        pbar->printProgressBox(i);
        usleep(100000);
    }
}