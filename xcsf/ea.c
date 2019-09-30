/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
 *
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
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "cl_set.h"    
#include "ea.h"
#include "sam.h"

CL *ea_select_parent(XCSF *xcsf, SET *set, double fit_sum);
void ea_subsume(XCSF *xcsf, CL *c, CL *c1p, CL *c2p, SET *set);

void ea(XCSF *xcsf, SET *set, SET *kset)
{
    // check if the evolutionary algorithm should be run
    if(set->size == 0 || xcsf->time - set_mean_time(xcsf, set) < xcsf->THETA_EA) {
        return;
    }
    set_times(xcsf, set);
    // select parents
    double fit_sum = set_total_fit(xcsf, set);
    CL *c1p = ea_select_parent(xcsf, set, fit_sum);
    CL *c2p = ea_select_parent(xcsf, set, fit_sum);

    for(int i = 0; i < xcsf->THETA_OFFSPRING/2; i++) {
        // create copies of parents
        CL *c1 = malloc(sizeof(CL));
        CL *c2 = malloc(sizeof(CL));
        cl_init(xcsf, c1, c1p->size, c1p->time);
        cl_init(xcsf, c2, c2p->size, c2p->time);
        cl_copy(xcsf, c1, c1p);
        cl_copy(xcsf, c2, c2p);
        // adapt mutation rates
        sam_adapt(xcsf, c1->mu);
        sam_adapt(xcsf, c2->mu);
        // apply evolutionary operators to offspring
        _Bool cmod = cl_crossover(xcsf, c1, c2);
        _Bool m1mod = cl_mutate(xcsf, c1);
        _Bool m2mod = cl_mutate(xcsf, c2); 
        // reduce offspring err, fit
        if(cmod) {
            c1->err = xcsf->ERR_REDUC * ((c1p->err + c2p->err)/2.0);
            c2->err = c1->err;
            c1->fit = c1p->fit / c1p->num;
            c2->fit = c2p->fit / c2p->num;
            c1->fit = xcsf->FIT_REDUC * ((c1->fit + c2->fit)/2.0);
            c2->fit = c1->fit;
        }
        else {
            c1->err = xcsf->ERR_REDUC * c1p->err;
            c2->err = xcsf->ERR_REDUC * c2p->err;
            c1->fit = xcsf->FIT_REDUC * (c1p->fit / c1p->num);
            c2->fit = xcsf->FIT_REDUC * (c2p->fit / c2p->num);
        }
        // add offspring to population
        if(xcsf->EA_SUBSUMPTION) {
            // c1 no crossover or mutation changes
            if(!cmod && !m1mod) {
                c1p->num++;
                xcsf->pset.num++;
                cl_free(xcsf, c1);      
            }
            else {
                ea_subsume(xcsf, c1, c1p, c2p, set);
            }
            // c2 no crossover or mutation changes
            if(!cmod && !m2mod) {
                c2p->num++;
                xcsf->pset.num++;
                cl_free(xcsf, c2);      
            }
            else {
                ea_subsume(xcsf, c2, c1p, c2p, set);
            }    
        }
        else {
            set_add(xcsf, &xcsf->pset, c1);
            set_add(xcsf, &xcsf->pset, c2);
        }
    }
    pop_enforce_limit(xcsf, kset);
}   

void ea_subsume(XCSF *xcsf, CL *c, CL *c1p, CL *c2p, SET *set)
{
    // check if either parent subsumes the offspring
    if(cl_subsumer(xcsf, c1p) && cl_general(xcsf, c1p, c)) {
        c1p->num++;
        xcsf->pset.num++;
        cl_free(xcsf, c);
    }
    else if(cl_subsumer(xcsf, c2p) && cl_general(xcsf, c2p, c)) {
        c2p->num++;
        xcsf->pset.num++;
        cl_free(xcsf, c);
    }
    // attempt to find a random subsumer from the set
    else {
        CLIST *candidates[set->size];
        int choices = 0;
        for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
            if(cl_subsumer(xcsf, iter->cl) && cl_general(xcsf, iter->cl, c)) {
                candidates[choices] = iter;
                choices++;
            }
        }
        // found
        if(choices > 0) {
            candidates[irand_uniform(0,choices)]->cl->num++;
            xcsf->pset.num++;
            cl_free(xcsf, c);
        }
        // if no subsumers are found the offspring is added to the population
        else {
            set_add(xcsf, &xcsf->pset, c);   
        }
    }
}

CL *ea_select_parent(XCSF *xcsf, SET *set, double fit_sum)
{
    (void)xcsf;
    // selects a classifier using roullete wheel selection with the fitness
    double p = rand_uniform(0,fit_sum);
    CLIST *iter = set->list;
    double sum = iter->cl->fit;
    while(p > sum) {
        iter = iter->next;
        sum += iter->cl->fit;
    }
    return iter->cl;
}