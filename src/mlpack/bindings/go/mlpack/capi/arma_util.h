/**
 * @file bindings/go/mlpack/capi/arma_util.h
 * @author Yasmine Dumouchl
 * @author Yashwant Singh
 *
 * Header file for cgo to call C functions from go.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_MLPACK_ARMAUTIL_H
#define MLPACK_BINDINGS_GO_MLPACK_ARMAUTIL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
void mlpackToArmaMat(void* params,
                     const char* identifier,
                     double* mat,
                     const size_t row,
                     const size_t col,
                     bool transpose);

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
void mlpackToArmaUmat(void* params,
                      const char* identifier,
                      double* mat,
                      const size_t row,
                      const size_t col);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaRow(void* params,
                     const char* identifier,
                     double* rowvec,
                     const size_t elem);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaUrow(void* params,
                      const char* identifier,
                      double* rowvec,
                      const size_t elem);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
void mlpackToArmaCol(void* params,
                     const char* identifier,
                     double* colvec,
                     const size_t elem);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
void mlpackToArmaUcol(void* params,
                      const char* identifier,
                      double* colvec,
                      const size_t elem);

/**
 * Return the memory pointer of an Armadillo mat object.
 */
void* mlpackArmaPtrMat(void* params, const char* identifier);

/**
 * Return the memory pointer of an Armadillo umat object.
 */
void* mlpackArmaPtrUmat(void* params, const char* identifier);

/**
 * Return the memory pointer of an Armadillo row object.
 */
void* mlpackArmaPtrRow(void* params, const char* identifier);

/**
 * Return the memory pointer of an Armadillo urow object.
 */
void* mlpackArmaPtrUrow(void* params, const char* identifier);

/**
 * Return the memory pointer of an Armadillo col object.
 */
void* mlpackArmaPtrCol(void* params, const char* identifier);

/**
 * Return the memory pointer of an Armadillo ucol object.
 */
void* mlpackArmaPtrUcol(void* params, const char* identifier);

/**
 * Return the number of rows in a Armadillo mat.
 */
int mlpackNumRowMat(void* params, const char* identifier);

/**
 * Return the number of columns in an Armadillo mat.
 */
int mlpackNumColMat(void* params, const char* identifier);

/**
 * Return the number of elements in an Armadillo mat.
 */
int mlpackNumElemMat(void* params, const char* identifier);

/**
 * Return the number of rows in an Armadillo umat.
 */
int mlpackNumRowUmat(void* params, const char* identifier);

/**
 * Return the number of columns in an Armadillo umat.
 */
int mlpackNumColUmat(void* params, const char* identifier);

/**
 * Return the number of elements in an Armadillo umat.
 */
int mlpackNumElemUmat(void* params, const char* identifier);

/**
 * Return the number of elements in an Armadillo row.
 */
int mlpackNumElemRow(void* params, const char* identifier);

/**
 * Return the number of elements in an Armadillo urow.
 */
int mlpackNumElemUrow(void* params, const char* identifier);

/**
 * Return the number of elements in an Armadillo col.
 */
int mlpackNumElemCol(void* params, const char* identifier);

/**
 * Return the number of elements in an Armadillo ucol.
 */
int mlpackNumElemUcol(void* params, const char* identifier);

/**
 * Call IO::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void mlpackToArmaMatWithInfo(void* params,
                             const char* identifier,
                             const bool* dimensions,
                             double* memptr,
                             const size_t rows,
                             const size_t cols);

/**
 * Get the number of elements in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoElements(void* params, const char* identifier);

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoRows(void* params, const char* identifier);

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoCols(void* params, const char* identifier);

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
void* mlpackArmaPtrMatWithInfoPtr(void* params, const char* identifier);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
