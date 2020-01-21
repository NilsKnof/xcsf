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
 * @file cond_ternary.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019.
 * @brief Ternary condition functions.
 * @details Binarises inputs.
 */ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "condition.h"
#include "cond_ternary.h"

#define P_DONTCARE 0.5 //!< Don't care probability in randomisation and covering
#define DONT_CARE '#' //!< Don't care symbol

/**
 * @brief Ternary condition data structure.
 */ 
typedef struct COND_TERNARY {
    char *string; //!< Ternary bitstring
    int len; //!< Length of the bitstring
} COND_TERNARY;

static void cond_ternary_rand(XCSF *xcsf, CL *c);

void cond_ternary_init(XCSF *xcsf, CL *c)
{
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    new->len = xcsf->num_x_vars * xcsf->COND_BITS;
    new->string = malloc(sizeof(char) * new->len);
    c->cond = new;     
    cond_ternary_rand(xcsf, c);
}

static void cond_ternary_rand(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    COND_TERNARY *cond = c->cond;
    for(int i = 0; i < cond->len; i++) {
        if(rand_uniform(0,1) < P_DONTCARE) {
            cond->string[i] = DONT_CARE;
        }
        else {
            if(rand_uniform(0,1) < 0.5) {
                cond->string[i] = '0';
            }
            else {
                cond->string[i] = '1';
            }
        }
    }
}

void cond_ternary_free(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    COND_TERNARY *cond = c->cond;
    free(cond->string);
    free(c->cond);
}

void cond_ternary_copy(XCSF *xcsf, CL *to, CL *from)
{
    (void)xcsf;
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    COND_TERNARY *from_cond = from->cond;
    new->len = from_cond->len;
    new->string = malloc(sizeof(char) * from_cond->len);
    memcpy(new->string, from_cond->string, sizeof(char) * from_cond->len);
    to->cond = new;
}                             

void cond_ternary_cover(XCSF *xcsf, CL *c, double *x)
{
    COND_TERNARY *cond = c->cond;
    char state[xcsf->COND_BITS];
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        float_to_binary(x[i], state, xcsf->COND_BITS);
        for(int b = 0; b < xcsf->COND_BITS; b++) {
            if(rand_uniform(0,1) < P_DONTCARE) {
                cond->string[i*xcsf->COND_BITS+b] = DONT_CARE;
            }
            else {
                cond->string[i*xcsf->COND_BITS+b] = state[b];
            }
        }
    }
}

void cond_ternary_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool cond_ternary_match(XCSF *xcsf, CL *c, double *x)
{
    COND_TERNARY *cond = c->cond;
    char state[xcsf->COND_BITS];
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        float_to_binary(x[i], state, xcsf->COND_BITS);
        for(int b = 0; b < xcsf->COND_BITS; b++) {
            char s = cond->string[i*xcsf->COND_BITS+b];
            if(s != DONT_CARE && s != state[b]) {
                c->m = false;
                return false;
            }
        }
    }
    c->m = true;
    return true;
}

_Bool cond_ternary_crossover(XCSF *xcsf, CL *c1, CL *c2) 
{
    COND_TERNARY *cond1 = c1->cond;
    COND_TERNARY *cond2 = c2->cond;
    _Bool changed = false;
    if(rand_uniform(0,1) < xcsf->P_CROSSOVER) {
        for(int i = 0; i < cond1->len; i++) {
            if(rand_uniform(0,1) < 0.5) {
                double tmp = cond1->string[i];
                cond1->string[i] = cond2->string[i];
                cond2->string[i] = tmp;
                changed = true;
            }
        }
    }
    return changed;
}

_Bool cond_ternary_mutate(XCSF *xcsf, CL *c)
{
    COND_TERNARY *cond = c->cond;
    _Bool changed = false;
    for(int i = 0; i < cond->len; i++) {
        if(rand_uniform(0,1) < xcsf->P_MUTATION) {
            if(cond->string[i] == DONT_CARE) {
                if(rand_uniform(0,1) < 0.5) {
                    cond->string[i] = '1';
                }
                else {
                    cond->string[i] = '0';
                }
            }
            else {
                cond->string[i] = DONT_CARE;
            }
            changed = true;
        }
    }
    return changed;
}

_Bool cond_ternary_general(XCSF *xcsf, CL *c1, CL *c2)
{
    // returns whether cond1 is more general than cond2
    (void)xcsf;
    COND_TERNARY *cond1 = c1->cond;
    COND_TERNARY *cond2 = c2->cond;
    _Bool general = false;
    for(int i = 0; i < cond1->len; i++) {
        if(cond1->string[i] != DONT_CARE && cond1->string[i] != cond2->string[i]) {
            return false;
        }
        else if(cond1->string[i] != cond2->string[i]) {
            general = true;
        }
    }
    return general;
}  

void cond_ternary_print(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    COND_TERNARY *cond = c->cond;
    printf("ternary:");
    for(int i = 0; i < cond->len; i++) {
        printf("%c", cond->string[i]);
    }
    printf("\n");
}

int cond_ternary_size(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    COND_TERNARY *cond = c->cond;
    return cond->len;
}

size_t cond_ternary_save(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    COND_TERNARY *cond = c->cond;
    s += fwrite(&cond->len, sizeof(int), 1, fp);
    s += fwrite(cond->string, sizeof(char), cond->len, fp);
    return s;
}

size_t cond_ternary_load(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    s += fread(&new->len, sizeof(int), 1, fp);
    new->string = malloc(sizeof(char) * new->len);
    s += fread(new->string, sizeof(char), new->len, fp);
    c->cond = new;
    return s;
}