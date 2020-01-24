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
 * @file env.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Built-in problem environment interface.
 */ 

#pragma once

void env_init(XCSF *xcsf, char **argv);

/**
 * @brief Built-in problem environment interface data structure.
 * @details Environment implementations must implement these functions.
 */ 
struct EnvVtbl {
    /**
     * @brief Returns whether the environment needs to be reset.
     * @param xcsf The XCSF data structure.
     * @return Whether the environment needs to be reset.
     */
    _Bool (*env_impl_isreset)(const XCSF *xcsf);
    /**
     * @brief Returns whether the environment is a multistep problem.
     * @param xcsf The XCSF data structure.
     * @return Whether the environment is multistep.
     */
    _Bool (*env_impl_multistep)(const XCSF *xcsf);
    /**
     * @brief Executes the specified action and returns the payoff.
     * @param xcsf The XCSF data structure.
     * @param action The action to perform.
     * @return The payoff from performing the action.
     */
    double (*env_impl_execute)(XCSF *xcsf, int action);
    /**
     * @brief Returns the maximum payoff value possible in the environment.
     * @param xcsf The XCSF data structure.
     * @return The maximum payoff.
     */
    double (*env_impl_max_payoff)(const XCSF *xcsf);
    /**
     * @brief Returns the current environment perceptions.
     * @param xcsf The XCSF data structure.
     * @return The current perceptions.
     */
    const double *(*env_impl_get_state)(XCSF *xcsf);
    /**
     * @brief Frees the environment.
     * @param xcsf The XCSF data structure.
     */
    void (*env_impl_free)(XCSF *xcsf);
    /**
     * @brief Resets the environment.
     * @param xcsf The XCSF data structure.
     */
    void (*env_impl_reset)(XCSF *xcsf);
};

static inline _Bool env_is_reset(const XCSF *xcsf) {
    return (*xcsf->env_vptr->env_impl_isreset)(xcsf);
}

static inline _Bool env_multistep(const XCSF *xcsf) {
    return (*xcsf->env_vptr->env_impl_multistep)(xcsf);
}

static inline double env_execute(XCSF *xcsf, int action) {
    return (*xcsf->env_vptr->env_impl_execute)(xcsf, action);
}

static inline double env_max_payoff(const XCSF *xcsf) {
    return (*xcsf->env_vptr->env_impl_max_payoff)(xcsf);
}

static inline const double *env_get_state(XCSF *xcsf) {
    return (*xcsf->env_vptr->env_impl_get_state)(xcsf);
}

static inline void env_free(XCSF *xcsf) {
    (*xcsf->env_vptr->env_impl_free)(xcsf);
}

static inline void env_reset(XCSF *xcsf) {
    (*xcsf->env_vptr->env_impl_reset)(xcsf);
}
