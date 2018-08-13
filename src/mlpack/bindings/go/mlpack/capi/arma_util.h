/**
 * @file arma_util.h
 * @author Yasmine Dumouchl
 *
 * Header file for cgo to call C functions from go.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_MLPACK_ARMA_UTIL_H
#define MLPACK_BINDINGS_GO_MLPACK_ARMA_UTIL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
extern void MLPACK_ToArma_mat(const char *identifier, const double mat[], int row, int col);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
extern void MLPACK_ToArma_row(const char *identifier, const double rowvec[], int elem);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
extern void MLPACK_ToArma_col(const char *identifier, const double colvec[], int elem);

/**
 * Return the memory pointer of an Armadillo mat object.
 */
extern void *MLPACK_ArmaPtr_mat(const char *identifier);

/**
 * Return the memory pointer of an Armadillo umat object.
 */
extern void *MLPACK_ArmaPtr_umat(const char *identifier);

/**
 * Return the memory pointer of an Armadillo row object.
 */
extern void *MLPACK_ArmaPtr_row(const char *identifier);

/**
 * Return the memory pointer of an Armadillo urow object.
 */
extern void *MLPACK_ArmaPtr_urow(const char *identifier);

/**
 * Return the memory pointer of an Armadillo col object.
 */
extern void *MLPACK_ArmaPtr_col(const char *identifier);

/**
 * Return the memory pointer of an Armadillo ucol object.
 */
extern void *MLPACK_ArmaPtr_ucol(const char *identifier);

/**
 * Return the number of rows in a Armadillo mat.
 */
extern int MLPACK_NumRow_mat(const char *identifier);

/**
 * Return the number of columns in an Armadillo mat.
 */
extern int MLPACK_NumCol_mat(const char *identifier);

/**
 * Return the number of elements in an Armadillo mat.
 */
extern int MLPACK_NumElem_mat(const char *identifier);

/**
 * Return the number of rows in an Armadillo umat.
 */
extern int MLPACK_NumRow_umat(const char *identifier);

/**
 * Return the number of columns in an Armadillo umat.
 */
extern int MLPACK_NumCol_umat(const char *identifier);

/**
 * Return the number of elements in an Armadillo umat.
 */
extern int MLPACK_NumElem_umat(const char *identifier);

/**
 * Return the number of elements in an Armadillo row.
 */
extern int MLPACK_NumElem_row(const char *identifier);

/**
 * Return the number of elements in an Armadillo urow.
 */
extern int MLPACK_NumElem_urow(const char *identifier);

/**
 * Return the number of elements in an Armadillo col.
 */
extern int MLPACK_NumElem_col(const char *identifier);

/**
 * Return the number of elements in an Armadillo ucol.
 */
extern int MLPACK_NumElem_ucol(const char *identifier);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
