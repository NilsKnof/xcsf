/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
          
/**
 * @file perf.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief System performance printing.
 */ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <errno.h>
#include "xcsf.h"
#include "cl.h"
#include "clset.h"
#include "perf.h"

/**
 * @brief Displays the current training performance.
 * @param xcsf The XCSF data structure.
 * @param error The current training error.
 * @param trial The number of learning trials executed.
 */
void disp_perf1(const XCSF *xcsf, double *error, int trial)
{
    disp_perf2(xcsf, error, NULL, trial);
}

/**
 * @brief Displays the current training and test performance.
 * @param xcsf The XCSF data structure.
 * @param error The current training error.
 * @param terror The current testing error.
 * @param trial The number of learning trials executed.
 */
void disp_perf2(const XCSF *xcsf, double *error, double *terror, int trial)
{
    if(trial % xcsf->PERF_TRIALS == 0 && trial > 0) {
        printf("%d %.5f ", trial, *error / xcsf->PERF_TRIALS);
        *error = 0;
        if(terror != NULL) {
            printf("%.5f ", *terror / xcsf->PERF_TRIALS);
            *terror = 0;
        }
        printf("%d", xcsf->pset.size);
        for(int i = 0; i < xcsf->SAM_NUM; i++) {
            printf(" %.5f", clset_mean_mut(xcsf, &xcsf->pset, i));
        }
        printf("\n");
        fflush(stdout);
    }
}
