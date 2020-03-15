
#ifndef SOES_H
#define SOES_H

#include <iostream>
#include <stdio.h>

class PBAR {

    public:

        PBAR(int);

        void printProgressBox(int);

        const char* filler = "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";

    private:

        int lenOfPBAR = 50;
        int current_progress = 0;
        int max_progress = 0;
        int current_lenOfPBAR = 0;
};

#endif