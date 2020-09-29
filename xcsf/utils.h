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
 * @file utils.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Utility functions for random number handling, etc.
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>

double
rand_normal(const double mu, const double sigma);

double
rand_uniform(const double min, const double max);

int
rand_uniform_int(const int min, const int max);

void
rand_init(void);

/**
 * @brief Returns a float clamped within the specified range.
 * @param a The value to be clamped.
 * @param min Minimum value.
 * @param max Maximum value.
 * @return The clamped number.
 */
static inline double
clamp(const double a, const double min, const double max)
{
    return (a < min) ? min : (a > max) ? max : a;
}

/**
 * @brief Returns an integer clamped within the specified range.
 * @param a The value to be clamped.
 * @param min Minimum value.
 * @param max Maximum value.
 * @return The clamped number.
 */
static inline int
clamp_int(const int a, const int min, const int max)
{
    return (a < min) ? min : (a > max) ? max : a;
}

/**
 * @brief Returns the index of the largest element in vector X.
 * @param X Vector with N elements.
 * @param N The number of elements in vector X.
 * @return The index of the largest element.
 */
static inline int
max_index(const double *X, const int N)
{
    if (N < 1) {
        printf("max_index() error: N < 1\n");
        exit(EXIT_FAILURE);
    }
    int max_i = 0;
    double max = X[0];
    for (int i = 1; i < N; ++i) {
        if (X[i] > max) {
            max_i = i;
            max = X[i];
        }
    }
    return max_i;
}
