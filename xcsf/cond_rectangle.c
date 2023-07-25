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
 * @file cond_rectangle.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2022.
 * @brief Hyperrectangle condition functions.
 */

#include "cond_rectangle.h"
#include "ea.h"
#include "sam.h"
#include "utils.h"

#define N_MU (1) //!< Number of hyperrectangle mutation rates

/**
 * @brief Self-adaptation method for mutating hyperrectangles.
 */
static const int MU_TYPE[N_MU] = { SAM_LOG_NORMAL };

/**
 * @brief Returns the relative distance to a hyperrectangle.
 * @details Distance is zero at the center; one on the border; and greater than
 * one outside of the hyperrectangle.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose hyperrectangle distance is to be computed.
 * @param [in] x Input to compute the relative distance.
 * @return The relative distance of an input to the hyperrectangle.
 */
static double
cond_rectangle_dist(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x)
{
    const struct CondRectangle *cond = c->cond;
    double dist = 0;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        const double d = fabs((x[i] - cond->b1[i]) / cond->b2[i]);
        if (d > dist) {
            dist = d;
        }
    }
    return dist;
}

/**
 * @brief Creates and initialises a hyperrectangle condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be initialised.
 */
void
cond_rectangle_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct CondRectangle *new = malloc(sizeof(struct CondRectangle));
    new->b1 = malloc(sizeof(double) * xcsf->x_dim);
    new->b2 = malloc(sizeof(double) * xcsf->x_dim);
    double temp = 0;
    const double spread_max = fabs(xcsf->cond->max - xcsf->cond->min);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        new->b1[i] = rand_uniform(xcsf->cond->min, xcsf->cond->max);
        new->b2[i] = rand_uniform(xcsf->cond->min, xcsf->cond->max);
    }
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        // csr: b1 = center, b2 = spread
        for (int i = 0; i < xcsf->x_dim; ++i) {
            new->b2[i] = rand_uniform(xcsf->cond->spread_min, spread_max);
        }
    } else if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MMR || xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MPR) {
        // MMR/OBR: b1 = lower, b2 = upper
        for (int i = 0; i < xcsf->x_dim; ++i) {
            if (new->b1[i] > new->b2[i]) {
                temp = new->b2[i];
                new->b2[i] = new->b1[i];
                new->b1[i] = temp;
            }
            if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MPR) {
                // MPR: b1 = lower, b2 = proportion
                new->b2[i] = (new->b2[i] - new->b1[i]) / (xcsf->cond->max - new->b1[i]);
            }
        }
    }
    new->mu = malloc(sizeof(double) * N_MU);
    sam_init(new->mu, N_MU, MU_TYPE);
    c->cond = new;
}


/**
 * @brief Converts an UBR condition to a new hyperrectangle condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be initialised.
 * @param [in] UBR Classifier whose condition is to be converted.
 */
void cond_rectangle_conv_init(const struct XCSF *xcsf, struct Cl *c, struct Cl *temp)
{
    struct CondRectangle *new = malloc(sizeof(struct CondRectangle));
    const struct CondRectangle *temp_cond = temp->cond;
    new->b1 = malloc(sizeof(double) * xcsf->x_dim);
    new->b2 = malloc(sizeof(double) * xcsf->x_dim);
    double b1 = 0, b2 = 0;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        b1 = fmin(temp_cond->b1[i], temp_cond->b2[i]);
        b2 = fmax(temp_cond->b1[i], temp_cond->b2[i]);
        if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
            // csr: b1 = center, b2 = spread
            new->b1[i] = b1 + ((b2 - b1) / 2);
            new->b2[i] = (b2 - b1) / 2;
        } else if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MMR) {
            // MMR/OBR: b1 = lower, b2 = upper
            new->b1[i] = b1;
            new->b2[i] = b2;
        } else if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MPR) {
            // MPR: b1 = lower, b2 = proportion
            new->b1[i] = b1;
            new->b2[i] = (b2 - b1) / (xcsf->cond->max - b1);
        }
    }
    new->mu = malloc(sizeof(double) * N_MU);
    sam_init(new->mu, N_MU, MU_TYPE);
    c->cond = new;
}

/**
 * @brief Frees the memory used by a hyperrectangle condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be freed.
 */
void
cond_rectangle_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondRectangle *cond = c->cond;
    free(cond->b1);
    free(cond->b2);
    free(cond->mu);
    free(c->cond);
}

/**
 * @brief Copies a hyperrectangle condition from one classifier to another.
 * @param [in] xcsf XCSF data structure.
 * @param [in] dest Destination classifier.
 * @param [in] src Source classifier.
 */
void
cond_rectangle_copy(const struct XCSF *xcsf, struct Cl *dest,
                    const struct Cl *src)
{
    struct CondRectangle *new = malloc(sizeof(struct CondRectangle));
    const struct CondRectangle *src_cond = src->cond;
    new->b1 = malloc(sizeof(double) * xcsf->x_dim);
    new->b2 = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    memcpy(new->b1, src_cond->b1, sizeof(double) * xcsf->x_dim);
    memcpy(new->b2, src_cond->b2, sizeof(double) * xcsf->x_dim);
    memcpy(new->mu, src_cond->mu, sizeof(double) * N_MU);
    dest->cond = new;
}

/**
 * @brief Generates a hyperrectangle that matches the current input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being covered.
 * @param [in] x Input state to cover.
 */
void
cond_rectangle_cover(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x)
{
    const struct CondRectangle *cond = c->cond;
    const double spread_max = fabs(xcsf->cond->max - xcsf->cond->min);
    // CSR
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            cond->b1[i] = x[i];
            cond->b2[i] = rand_uniform(xcsf->cond->spread_min, spread_max);
        }
    // OBR, UBR & MPR
    } else {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            const double r1 = rand_uniform(xcsf->cond->spread_min, spread_max);
            const double r2 = rand_uniform(xcsf->cond->spread_min, spread_max);
            cond->b1[i] = (x[i] - r1 < xcsf->cond->min) ? xcsf->cond->min : x[i] - r1;
            cond->b2[i] = (x[i] + r2 > xcsf->cond->max) ? xcsf->cond->max : x[i] + r2;
            // random swap for UBR
            if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_UBR && rand_uniform(0, 1) < 0.5) {
                double temp = cond->b1[i];
                cond->b1[i] = cond->b2[i];
                cond->b2[i] = temp;
            // MPR
            } else if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MPR) {
                cond->b2[i] = (cond->b2[i] - cond->b1[i]) / (xcsf->cond->max - cond->b1[i]);
            }
        }
    }
}

/**
 * @brief Updates a hyperrectangle, sliding the centers towards the mean input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
cond_rectangle_update(const struct XCSF *xcsf, const struct Cl *c,
                      const double *x, const double *y)
{
    (void) y;
    if (xcsf->cond->eta <= 0) return;
    const struct CondRectangle *cond = c->cond;
    // CSR
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            cond->b1[i] += xcsf->cond->eta * (x[i] - cond->b1[i]);
        }
    } else {
        double lb = 0, ub = 0, offset = 0;
        for (int i = 0; i < xcsf->x_dim; ++i) {
            // UBR
            if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_UBR) {
                lb = fmin(cond->b1[i], cond->b2[i]);
                ub = fmax(cond->b1[i], cond->b2[i]);
            // MMR
            } else if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MMR) {
                lb = cond->b1[i];
                ub = cond->b2[i];
            //MPR
            } else {
                lb = cond->b1[i];
                ub = lb + (cond->b2[i] * (xcsf->cond->max - lb));
            }
            offset = xcsf->cond->eta * (x[i] - (lb + ((ub - lb) / 2)));
            // MPR
            if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MPR) {
                ub += offset;
                cond->b1[i] += offset;
                cond->b2[i] = (ub - cond->b1[i]) / (xcsf->cond->max - cond->b1[i]);
            // UBR & OBR
            } else {
                cond->b1[i] += offset;
                cond->b2[i] += offset;
            }
        }
    }
}

/**
 * @brief Calculates whether a hyperrectangle condition matches an input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition to match.
 * @param [in] x Input state.
 * @return Whether the hyperrectangle condition matches the input.
 */
bool
cond_rectangle_match(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x)
{
    const struct CondRectangle *cond = c->cond;
    // CSR
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        return (cond_rectangle_dist(xcsf, c, x) <= 1);
    }
    for (int i = 0; i < xcsf->x_dim; ++i) {
        double lb = 0, ub = 0;
        // UBR
        if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_UBR) {
            lb = fmin(cond->b1[i], cond->b2[i]);
            ub = fmax(cond->b1[i], cond->b2[i]);
        // MMR
        } else if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MMR) {
            lb = cond->b1[i];
            ub = cond->b2[i];
        // MPR
        } else {
            lb = cond->b1[i];
            ub = lb + (cond->b2[i] * (xcsf->cond->max - lb));
        }
        lb = (lb < xcsf->cond->min) ? xcsf->cond->min : lb;
        ub = (ub > xcsf->cond->max) ? xcsf->cond->max : ub;
        if (x[i] < lb || x[i] > ub) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Performs uniform crossover with two hyperrectangle conditions.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 First classifier whose condition is being crossed.
 * @param [in] c2 Second classifier whose condition is being crossed.
 * @return Whether any alterations were made.
 */
bool
cond_rectangle_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                         const struct Cl *c2)
{
    const struct CondRectangle *cond1 = c1->cond;
    const struct CondRectangle *cond2 = c2->cond;
    bool changed = false;
    if (rand_uniform(0, 1) < xcsf->ea->p_crossover) {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            if (rand_uniform(0, 1) < 0.5) {
                const double tmp = cond1->b1[i];
                cond1->b1[i] = cond2->b1[i];
                cond2->b1[i] = tmp;
                changed = true;
            }
            if (rand_uniform(0, 1) < 0.5) {
                const double tmp = cond1->b2[i];
                cond1->b2[i] = cond2->b2[i];
                cond2->b2[i] = tmp;
                changed = true;
            }
        }
    }
    return changed;
}

/**
 * @brief Mutates a hyperrectangle condition with the self-adaptive rate.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being mutated.
 * @return Whether any alterations were made.
 */
bool
cond_rectangle_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    bool changed = false;
    const struct CondRectangle *cond = c->cond;
    double *b1 = cond->b1;
    double *b2 = cond->b2;
    sam_adapt(cond->mu, N_MU, MU_TYPE);
    // MPR
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MPR) {
        double ub;
        for (int i = 0; i < xcsf->x_dim; ++i) {
            ub = b1[i] + (b2[i] * (xcsf->cond->max - b1[i]));
            double orig = b1[i];
            b1[i] += rand_normal(0, cond->mu[0]);
            b1[i] = clamp(b1[i], xcsf->cond->min, xcsf->cond->max);
            if (orig != b1[i]) {
                changed = true;
            }
            orig = b2[i];
            // changing b1[i] also changes b2[i]
            b2[i] = (ub - b1[i]) / (xcsf->cond->max - b1[i]);
            b2[i] += rand_normal(0, cond->mu[0]);
            b2[i] = clamp(b2[i], 0, 1);
            if (orig != b2[i]) {
                changed = true;
            }
        }
    } else {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            double orig = b1[i];
            b1[i] += rand_normal(0, cond->mu[0]);
            b1[i] = clamp(b1[i], xcsf->cond->min, xcsf->cond->max);
            if (orig != b1[i]) {
                changed = true;
            }
            orig = b2[i];
            b2[i] += rand_normal(0, cond->mu[0]);
            // CSR
            if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
                const double spread_max = fabs(xcsf->cond->max - xcsf->cond->min);
                // allowing maximal general intervals
                b2[i] = clamp(b2[i], 0, spread_max);
            // MMR & UBR
            } else {
                b2[i] = clamp(b2[i], xcsf->cond->min, xcsf->cond->max);
                // keep ordering constraint
                if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MMR && b2[i] < b1[i]) {
                    double temp = b1[i];
                    b1[i] = b2[i];
                    b2[i] = temp;
                }
            }
            if (orig != b2[i]) {
                changed = true;
            }
        }
    }
    return changed;
}

/**
 * @brief Returns whether classifier c1 has a condition more general than c2.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 Classifier whose condition is tested to be more general.
 * @param [in] c2 Classifier whose condition is tested to be more specific.
 * @return Whether the hyperrectangle condition of c1 is more general than c2.
 */
bool
cond_rectangle_general(const struct XCSF *xcsf, const struct Cl *c1,
                       const struct Cl *c2)
{
    const struct CondRectangle *cond1 = c1->cond;
    const struct CondRectangle *cond2 = c2->cond;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        double l1 = 0, l2 = 0, u1 = 0, u2 = 0;
        if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
            l1 = cond1->b1[i] - cond1->b2[i];
            l2 = cond2->b1[i] - cond2->b2[i];
            u1 = cond1->b1[i] + cond1->b2[i];
            u2 = cond2->b1[i] + cond2->b2[i];
        } else if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_UBR) {
            l1 = fmin(cond1->b1[i], cond1->b2[i]);
            l2 = fmin(cond2->b1[i], cond2->b2[i]);
            u1 = fmax(cond1->b1[i], cond1->b2[i]);
            u2 = fmax(cond2->b1[i], cond2->b2[i]);
        } else if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_MMR) {
            l1 = cond1->b1[i];
            l2 = cond2->b1[i];
            u1 = cond1->b2[i];
            u2 = cond2->b2[i];
        } else { //MPR
            l1 = cond1->b1[i];
            l2 = cond2->b1[i];
            u1 = l1 + (cond1->b2[i] * (xcsf->cond->max - l1));
            u2 = l2 + (cond2->b2[i] * (xcsf->cond->max - l2));
        }
        if (l1 > l2 || u1 < u2) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Prints a hyperrectangle condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be printed.
 */
void
cond_rectangle_print(const struct XCSF *xcsf, const struct Cl *c)
{
    char *json_str = cond_rectangle_json_export(xcsf, c);
    printf("%s\n", json_str);
    free(json_str);
}

/**
 * @brief Returns the size of a hyperrectangle condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition size to return.
 * @return The length of the input dimension.
 */
double
cond_rectangle_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) c;
    return xcsf->x_dim;
}

/**
 * @brief Writes a hyperrectangle condition to a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
cond_rectangle_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    size_t s = 0;
    const struct CondRectangle *cond = c->cond;
    s += fwrite(cond->b1, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->b2, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->mu, sizeof(double), N_MU, fp);
    return s;
}

/**
 * @brief Reads a hyperrectangle condition from a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
cond_rectangle_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    size_t s = 0;
    struct CondRectangle *new = malloc(sizeof(struct CondRectangle));
    new->b1 = malloc(sizeof(double) * xcsf->x_dim);
    new->b2 = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    s += fread(new->b1, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->b2, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->cond = new;
    return s;
}

/**
 * @brief Returns a json formatted string representation of a hyperrectangle.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be returned.
 * @return String encoded in json format.
 */
char *
cond_rectangle_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct CondRectangle *cond = c->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON *b1 = cJSON_CreateDoubleArray(cond->b1, xcsf->x_dim);
    cJSON *b2 = cJSON_CreateDoubleArray(cond->b2, xcsf->x_dim);
    cJSON *mutation = cJSON_CreateDoubleArray(cond->mu, N_MU);
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        cJSON_AddStringToObject(json, "type", "hyperrectangle_csr");
        cJSON_AddItemToObject(json, "center", b1);
        cJSON_AddItemToObject(json, "spread", b2);
    } else {
        cJSON_AddStringToObject(json, "type", "hyperrectangle_ubr");
        cJSON_AddItemToObject(json, "bound1", b1);
        cJSON_AddItemToObject(json, "bound2", b2);
    }
    cJSON_AddItemToObject(json, "mutation", mutation);
    char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Creates a hyperrectangle from a cJSON object.
 * @param [in] xcsf The XCSF data structure.
 * @param [in,out] c The classifier to initialise.
 * @param [in] json cJSON object.
 */
void
cond_rectangle_json_import(const struct XCSF *xcsf, struct Cl *c,
                           const cJSON *json)
{
    struct CondRectangle *cond = c->cond;
    bool csr = false;
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        csr = true;
    }
    const char *b1_name = csr ? "center" : "bound1";
    const char *b2_name = csr ? "spread" : "bound2";
    const cJSON *item = cJSON_GetObjectItem(json, b1_name);
    if (item != NULL && cJSON_IsArray(item)) {
        if (cJSON_GetArraySize(item) == xcsf->x_dim) {
            for (int i = 0; i < xcsf->x_dim; ++i) {
                const cJSON *item_i = cJSON_GetArrayItem(item, i);
                cond->b1[i] = item_i->valuedouble;
            }
        } else {
            printf("Import error: %s length mismatch\n", b1_name);
            exit(EXIT_FAILURE);
        }
    }
    item = cJSON_GetObjectItem(json, b2_name);
    if (item != NULL && cJSON_IsArray(item)) {
        if (cJSON_GetArraySize(item) == xcsf->x_dim) {
            for (int i = 0; i < xcsf->x_dim; ++i) {
                const cJSON *item_i = cJSON_GetArrayItem(item, i);
                cond->b2[i] = item_i->valuedouble;
            }
        } else {
            printf("Import error: %s length mismatch\n", b2_name);
            exit(EXIT_FAILURE);
        }
    }
    sam_json_import(cond->mu, N_MU, json);
}
