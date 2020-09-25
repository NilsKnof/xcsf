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
 * @file cond_gp.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Tree GP condition functions.
 */

#pragma once

#include "condition.h"
#include "gp.h"
#include "xcsf.h"

/**
 * @brief Tree GP condition data structure.
 */
struct COND_GP {
    struct GP_TREE gp; //!< GP tree
};

_Bool
cond_gp_crossover(const struct XCSF *xcsf, const struct CL *c1,
                  const struct CL *c2);

_Bool
cond_gp_general(const struct XCSF *xcsf, const struct CL *c1,
                const struct CL *c2);

_Bool
cond_gp_match(const struct XCSF *xcsf, const struct CL *c, const double *x);

_Bool
cond_gp_mutate(const struct XCSF *xcsf, const struct CL *c);

void
cond_gp_copy(const struct XCSF *xcsf, struct CL *dest, const struct CL *src);

void
cond_gp_cover(const struct XCSF *xcsf, const struct CL *c, const double *x);

void
cond_gp_free(const struct XCSF *xcsf, const struct CL *c);

void
cond_gp_init(const struct XCSF *xcsf, struct CL *c);

void
cond_gp_print(const struct XCSF *xcsf, const struct CL *c);

void
cond_gp_update(const struct XCSF *xcsf, const struct CL *c, const double *x,
               const double *y);

double
cond_gp_size(const struct XCSF *xcsf, const struct CL *c);

size_t
cond_gp_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp);

size_t
cond_gp_load(const struct XCSF *xcsf, struct CL *c, FILE *fp);

/**
 * @brief Tree GP condition implemented functions.
 */
static struct CondVtbl const cond_gp_vtbl = {
    &cond_gp_crossover, &cond_gp_general, &cond_gp_match, &cond_gp_mutate,
    &cond_gp_copy,      &cond_gp_cover,   &cond_gp_free,  &cond_gp_init,
    &cond_gp_print,     &cond_gp_update,  &cond_gp_size,  &cond_gp_save,
    &cond_gp_load
};
