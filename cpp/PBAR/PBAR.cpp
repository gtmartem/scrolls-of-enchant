#include "PBAR.h"

PBAR::PBAR(int max) {
    max_progress = max;
}

void PBAR::printProgressBox(int upd_progress) {
    current_progress = upd_progress;
    int space = (int)(((double)current_progress/(double)max_progress * (double)lenOfPBAR));
    printf("\r%d%%: [%.*s%*s]", current_progress, space, filler, 
            (int)(lenOfPBAR - space), "");
    fflush(stdout);
}