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
 * @file config.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Configuration file handling functions.
 */ 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <errno.h>
#include "xcsf.h"
#include "gp.h"
#include "config.h"
#include "loss.h"

#ifdef PARALLEL
#include <omp.h>
#endif

#define ARRAY_DELIM "," //!< Delimeter for config arrays
#define MAXLEN 127 //!< Maximum config file line length to read
#define BASE 10 //!< Decimal numbers

/**
 * @brief Configuration file parameter data structure.
 */ 
typedef struct PARAM_LIST {
    char *name; //!< Parameter name
    char *value; //!< Parameter value
    struct PARAM_LIST *next; //!< Pointer to the next parameter
} PARAM_LIST;

static PARAM_LIST *head; //!< Linked list of config file parameters

static char *config_getvalue(const char *name);
static int config_get_ints(const char *name, int *value);
static void config_newnvpair(const char *config);
static void config_process(const char *configline);
static void config_read(const char *filename);
static void config_tidyup();
static void config_trim(char *s);
static void params_cl_action(XCSF *xcsf);
static void params_cl_condition(XCSF *xcsf);
static void params_cl_general(XCSF *xcsf);
static void params_cl_prediction(XCSF *xcsf);
static void params_ea(XCSF *xcsf);
static void params_general(XCSF *xcsf);
static void params_multistep(XCSF *xcsf);
static void params_subsumption(XCSF *xcsf);

/**
 * @brief Initialises global constants and reads the specified configuration file.
 * @param xcsf The XCSF data structure.
 * @param filename The name of the config file to read.
 */
void config_init(XCSF *xcsf, const char *filename)
{
    config_read(filename);
    // initialise parameters
    params_general(xcsf);
    params_multistep(xcsf);
    params_ea(xcsf);
    params_subsumption(xcsf);
    params_cl_general(xcsf);
    params_cl_condition(xcsf);
    params_cl_action(xcsf);
    params_cl_prediction(xcsf);
    // initialise (shared) tree-GP constants
    tree_init_cons(xcsf);
    // initialise loss/error function
    loss_set_func(xcsf);
    // clean up
    config_tidyup();
}

/**
 * @brief Frees all global constants.
 * @param xcsf The XCSF data structure.
 */
void config_free(const XCSF *xcsf)
{
    tree_free_cons(xcsf);
}

/**
 * @brief Initialises general system parameters.
 * @param xcsf The XCSF data structure.
 */
static void params_general(XCSF *xcsf)
{
    char *end;
    xcsf->OMP_NUM_THREADS = strtoimax(config_getvalue("OMP_NUM_THREADS"), &end, BASE);
#ifdef PARALLEL
    omp_set_num_threads(xcsf->OMP_NUM_THREADS);
#endif
    xcsf->POP_SIZE = strtoimax(config_getvalue("POP_SIZE"), &end, BASE);
    xcsf->MAX_TRIALS = strtoimax(config_getvalue("MAX_TRIALS"), &end, BASE);
    xcsf->POP_INIT = false;
    if(strncmp(config_getvalue("POP_INIT"), "true", 4) == 0) {
        xcsf->POP_INIT = true;
    }
    xcsf->PERF_TRIALS = strtoimax(config_getvalue("PERF_TRIALS"), &end, BASE);
    xcsf->LOSS_FUNC = strtoimax(config_getvalue("LOSS_FUNC"), &end, BASE);
}

/**
 * @brief Initialises multi-step parameters.
 * @param xcsf The XCSF data structure.
 */
static void params_multistep(XCSF *xcsf)
{
    char *end;
    xcsf->TELETRANSPORTATION = strtoimax(config_getvalue("TELETRANSPORTATION"), &end, BASE);
    xcsf->GAMMA = atof(config_getvalue("GAMMA"));
    xcsf->P_EXPLORE = atof(config_getvalue("P_EXPLORE"));
}

/**
 * @brief Initialises evolutionary algorithm parameters.
 * @param xcsf The XCSF data structure.
 */
static void params_ea(XCSF *xcsf)
{
    char *end;
    xcsf->EA_SELECT_TYPE = strtoimax(config_getvalue("EA_SELECT_TYPE"), &end, BASE);
    xcsf->EA_SELECT_SIZE = atof(config_getvalue("EA_SELECT_SIZE"));
    xcsf->THETA_EA = atof(config_getvalue("THETA_EA"));
    xcsf->LAMBDA = strtoimax(config_getvalue("LAMBDA"), &end, BASE);
    xcsf->P_CROSSOVER = atof(config_getvalue("P_CROSSOVER"));
    xcsf->SAM_TYPE = strtoimax(config_getvalue("SAM_TYPE"), &end, BASE);
}

/**
 * @brief Initialises subsumption parameters.
 * @param xcsf The XCSF data structure.
 */
static void params_subsumption(XCSF *xcsf)
{
    char *end;
    xcsf->EA_SUBSUMPTION = false;
    if(strncmp(config_getvalue("EA_SUBSUMPTION"), "true", 4) == 0) {
        xcsf->EA_SUBSUMPTION = true;
    }
    xcsf->SET_SUBSUMPTION = false;
    if(strncmp(config_getvalue("SET_SUBSUMPTION"), "true", 4) == 0) {
        xcsf->SET_SUBSUMPTION = true;
    }
    xcsf->THETA_SUB = strtoimax(config_getvalue("THETA_SUB"), &end, BASE);
}

/**
 * @brief Initialises general classifier parameters.
 * @param xcsf The XCSF data structure.
 */
static void params_cl_general(XCSF *xcsf)
{
    char *end;
    xcsf->ALPHA = atof(config_getvalue("ALPHA"));
    xcsf->BETA = atof(config_getvalue("BETA"));
    xcsf->DELTA = atof(config_getvalue("DELTA"));
    xcsf->NU = atof(config_getvalue("NU"));
    xcsf->THETA_DEL = strtoimax(config_getvalue("THETA_DEL"), &end, BASE);
    xcsf->INIT_FITNESS = atof(config_getvalue("INIT_FITNESS"));
    xcsf->INIT_ERROR = atof(config_getvalue("INIT_ERROR"));
    xcsf->ERR_REDUC = atof(config_getvalue("ERR_REDUC"));
    xcsf->FIT_REDUC = atof(config_getvalue("FIT_REDUC"));
    xcsf->EPS_0 = atof(config_getvalue("EPS_0"));
    xcsf->M_PROBATION = strtoimax(config_getvalue("M_PROBATION"), &end, BASE);
}

/**
 * @brief Initialises classifier condition parameters.
 * @param xcsf The XCSF data structure.
 */
static void params_cl_condition(XCSF *xcsf)
{
    char *end;
    xcsf->COND_MIN = atof(config_getvalue("COND_MIN"));
    xcsf->COND_MAX = atof(config_getvalue("COND_MAX"));
    xcsf->COND_TYPE = strtoimax(config_getvalue("COND_TYPE"), &end, BASE);
    xcsf->COND_SMIN = atof(config_getvalue("COND_SMIN"));
    xcsf->COND_BITS = strtoimax(config_getvalue("COND_BITS"), &end, BASE);
    xcsf->COND_ETA = atof(config_getvalue("COND_ETA"));
    xcsf->GP_NUM_CONS = strtoimax(config_getvalue("GP_NUM_CONS"), &end, BASE);
    xcsf->GP_INIT_DEPTH = strtoimax(config_getvalue("GP_INIT_DEPTH"), &end, BASE);
    xcsf->DGP_NUM_NODES = strtoimax(config_getvalue("DGP_NUM_NODES"), &end, BASE);
    xcsf->RESET_STATES = false;
    if(strncmp(config_getvalue("RESET_STATES"), "true", 4) == 0) {
        xcsf->RESET_STATES = true;
    }
    xcsf->MAX_K = strtoimax(config_getvalue("MAX_K"), &end, BASE);
    xcsf->MAX_T = strtoimax(config_getvalue("MAX_T"), &end, BASE);
    xcsf->MAX_NEURON_MOD = strtoimax(config_getvalue("MAX_NEURON_MOD"), &end, BASE);
    xcsf->COND_EVOLVE_WEIGHTS = false;
    if(strncmp(config_getvalue("COND_EVOLVE_WEIGHTS"), "true", 4) == 0) {
        xcsf->COND_EVOLVE_WEIGHTS = true;
    }
    xcsf->COND_EVOLVE_NEURONS = false;
    if(strncmp(config_getvalue("COND_EVOLVE_NEURONS"), "true", 4) == 0) {
        xcsf->COND_EVOLVE_NEURONS = true;
    }
    xcsf->COND_EVOLVE_FUNCTIONS = false;
    if(strncmp(config_getvalue("COND_EVOLVE_FUNCTIONS"), "true", 4) == 0) {
        xcsf->COND_EVOLVE_FUNCTIONS = true;
    }
    memset(xcsf->COND_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
    config_get_ints("COND_NUM_NEURONS", xcsf->COND_NUM_NEURONS);
    memset(xcsf->COND_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
    config_get_ints("COND_MAX_NEURONS", xcsf->COND_MAX_NEURONS);
    xcsf->COND_OUTPUT_ACTIVATION = strtoimax(config_getvalue("COND_OUTPUT_ACTIVATION"), &end, BASE);
    xcsf->COND_HIDDEN_ACTIVATION = strtoimax(config_getvalue("COND_HIDDEN_ACTIVATION"), &end, BASE);
}

/**
 * @brief Initialises classifier action parameters.
 * @param xcsf The XCSF data structure.
 */
static void params_cl_action(XCSF *xcsf)
{
    char *end;
    xcsf->ACT_TYPE = strtoimax(config_getvalue("ACT_TYPE"), &end, BASE);
}

/**
 * @brief Initialises classifier prediction parameters.
 * @param xcsf The XCSF data structure.
 */
static void params_cl_prediction(XCSF *xcsf)
{
    char *end;
    xcsf->PRED_TYPE = strtoimax(config_getvalue("PRED_TYPE"), &end, BASE);
    xcsf->PRED_ETA = atof(config_getvalue("PRED_ETA"));
    xcsf->PRED_RESET = false;
    if(strncmp(config_getvalue("PRED_RESET"), "true", 4) == 0) {
        xcsf->PRED_RESET = true;
    }
    xcsf->PRED_X0 = atof(config_getvalue("PRED_X0"));
    xcsf->PRED_RLS_SCALE_FACTOR = atof(config_getvalue("PRED_RLS_SCALE_FACTOR"));
    xcsf->PRED_RLS_LAMBDA = atof(config_getvalue("PRED_RLS_LAMBDA"));
    xcsf->PRED_EVOLVE_WEIGHTS = false;
    if(strncmp(config_getvalue("PRED_EVOLVE_WEIGHTS"), "true", 4) == 0) {
        xcsf->PRED_EVOLVE_WEIGHTS = true;
    }
    xcsf->PRED_EVOLVE_NEURONS = false;
    if(strncmp(config_getvalue("PRED_EVOLVE_NEURONS"), "true", 4) == 0) {
        xcsf->PRED_EVOLVE_NEURONS = true;
    }
    xcsf->PRED_EVOLVE_FUNCTIONS = false;
    if(strncmp(config_getvalue("PRED_EVOLVE_FUNCTIONS"), "true", 4) == 0) {
        xcsf->PRED_EVOLVE_FUNCTIONS = true;
    }
    xcsf->PRED_EVOLVE_ETA = false;
    if(strncmp(config_getvalue("PRED_EVOLVE_ETA"), "true", 4) == 0) {
        xcsf->PRED_EVOLVE_ETA = true;
    }
    xcsf->PRED_SGD_WEIGHTS = false;
    if(strncmp(config_getvalue("PRED_SGD_WEIGHTS"), "true", 4) == 0) {
        xcsf->PRED_SGD_WEIGHTS = true;
    }
    xcsf->PRED_MOMENTUM = atof(config_getvalue("PRED_MOMENTUM"));
    memset(xcsf->PRED_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
    config_get_ints("PRED_NUM_NEURONS", xcsf->PRED_NUM_NEURONS);
    memset(xcsf->PRED_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
    config_get_ints("PRED_MAX_NEURONS", xcsf->PRED_MAX_NEURONS);
    xcsf->PRED_OUTPUT_ACTIVATION = strtoimax(config_getvalue("PRED_OUTPUT_ACTIVATION"), &end, BASE);
    xcsf->PRED_HIDDEN_ACTIVATION = strtoimax(config_getvalue("PRED_HIDDEN_ACTIVATION"), &end, BASE);
}

/**
 * @brief Reads a csv list of ints into an array.
 * @param name The name of the parameter.
 * @param value An integer array (set by this function).
 * @return The number of values read into the array.
 */
static int config_get_ints(const char *name, int *value)
{
    int num = 0;
    char *end;
    char *saveptr;
    char *val = config_getvalue(name);
    const char *ptok = strtok_r(val, ARRAY_DELIM, &saveptr);
    while(ptok != NULL) {
        value[num] = strtoimax(ptok, &end, BASE);
        ptok = strtok_r(NULL, ARRAY_DELIM, &saveptr);
        num++;
    }
    return num;
}

/**
 * @brief Removes tabs/spaces/lf/cr
 * @param s The line to trim.
 */
static void config_trim(char *s) {
    const char *d = s;
    do {
        while(*d == ' ' || *d == '\t' || *d == '\n' || *d == '\r') {
            d++;
        }
    } while((*s++ = *d++));
}

/**
 * @brief Adds a parameter to the list.
 * @param config The parameter to add.
 */
static void config_newnvpair(const char *config) {
    // first pair
    if(head == NULL) {
        head = malloc(sizeof(PARAM_LIST));
        head->next = NULL;
    }
    // other pairs
    else {
        PARAM_LIST *new = malloc(sizeof(PARAM_LIST));
        new->next = head;
        head = new;
    }
    // get length of name
    size_t namelen = 0; // length of name
    _Bool err = true;
    for(namelen = 0; namelen < strnlen(config, MAXLEN); namelen++) {
        if(config[namelen] == '=') {
            err = false;
            break;
        }
    }
    // no = found
    if(err) {
        printf("error reading config: no '=' found\n");
        exit(EXIT_FAILURE);
    }
    // get name
    char *name = malloc(namelen+1);
    for(size_t i = 0; i < namelen; i++) {
        name[i] = config[i];
    }
    name[namelen] = '\0';
    // get value
    size_t valuelen = strnlen(config,MAXLEN)-namelen; // length of value
    char *value = malloc(valuelen+1);
    for(size_t i = 0; i < valuelen; i++) {
        value[i] = config[namelen+1+i];
    }
    value[valuelen] = '\0';
    // add pair
    head->name = name;
    head->value = value;
}

/**
 * @brief Returns the value of a specified parameter from the list.
 * @param name The name of the parameter.
 * @return The value of the parameter.
 */
static char *config_getvalue(const char *name) {
    char *result = NULL;
    for(const PARAM_LIST *iter = head; iter != NULL; iter = iter->next) {
        if(strcmp(name, iter->name) == 0) {
            result = iter->value;
            break;
        }
    }
    return result;
}

/**
 * @brief Parses a line of the config file and adds to the list.
 * @param configline A single line of the configuration file.
 */
static void config_process(const char *configline) {
    // ignore empty lines
    if(strnlen(configline, MAXLEN) == 0) {
        return;
    }
    // lines starting with # are comments
    if(configline[0] == '#') {
        return; 
    }
    // remove anything after #
    char *ptr = strchr(configline, '#');
    if(ptr != NULL) {
        *ptr = '\0';
    }
    config_newnvpair(configline);
}

/**
 * @brief Reads the specified configuration file.
 * @param filename The name of the configuration file.
 */
static void config_read(const char *filename) {
    FILE *f = fopen(filename, "rt");
    if(f == NULL) {
        printf("ERROR: cannot open %s\n", filename);
        return;
    }
    char buff[MAXLEN];
    head = NULL;
    while(!feof(f)) {
        if(fgets(buff, MAXLEN-2, f) == NULL) {
            break;
        }
        config_trim(buff);
        config_process(buff);
    }
    fclose(f);
}

/**
 * @brief Frees the config file parameter list.
 */
static void config_tidyup()
{ 
    PARAM_LIST *iter = head;
    while(iter != NULL) {
        free(head->value);
        free(head->name);
        head = iter->next;
        free(iter);
        iter = head;
    }    
    head = NULL;
}
