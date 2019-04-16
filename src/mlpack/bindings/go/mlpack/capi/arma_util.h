/**
 * @file arma_util.h
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
 * Pass Gonum Dense poconst size_t er and wrap an Armadillo mat around it.
 */
void mlpackToArmaMat(const char* identifier,
                            double* mat,
                            const size_t row,
                            const size_t col);

/**
 * Pass Gonum Dense poconst size_t er and wrap an Armadillo mat around it.
 */
void mlpackToArmaUmat(const char* identifier,
                             double* mat,
                             const size_t row,
                             const size_t col);

/**
 * Pass Gonum VecDense poconst size_t er and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaRow(const char* identifier,
                            double* rowvec,
                            const size_t elem);

/**
 * Pass Gonum VecDense poconst size_t er and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaUrow(const char* identifier,
                             double* rowvec,
                             const size_t elem);

/**
 * Pass Gonum VecDense poconst size_t er and wrap an Armadillo colvec around it.
 */
void mlpackToArmaCol(const char* identifier,
                            double* colvec,
                            const size_t elem);

/**
 * Pass Gonum VecDense poconst size_t er and wrap an Armadillo colvec around it.
 */
void mlpackToArmaUcol(const char* identifier,
                             double* colvec,
                             const size_t elem);

/**
 * Return the memory poconst size_t er of an Armadillo mat object.
 */
void* mlpackArmaPtrMat(const char* identifier);

/**
 * Return the memory poconst size_t er of an Armadillo umat object.
 */
void* mlpackArmaPtrUmat(const char* identifier);

/**
 * Return the memory poconst size_t er of an Armadillo row object.
 */
void* mlpackArmaPtrRow(const char* identifier);

/**
 * Return the memory poconst size_t er of an Armadillo urow object.
 */
void* mlpackArmaPtrUrow(const char* identifier);

/**
 * Return the memory poconst size_t er of an Armadillo col object.
 */
void* mlpackArmaPtrCol(const char* identifier);

/**
 * Return the memory poconst size_t er of an Armadillo ucol object.
 */
void* mlpackArmaPtrUcol(const char* identifier);

/**
 * Return the number of rows in a Armadillo mat.
 */
int mlpackNumRowMat(const char* identifier);

/**
 * Return the number of columns in an Armadillo mat.
 */
int mlpackNumColMat(const char* identifier);

/**
 * Return the number of elements in an Armadillo mat.
 */
int mlpackNumElemMat(const char* identifier);

/**
 * Return the number of rows in an Armadillo umat.
 */
int mlpackNumRowUmat(const char* identifier);

/**
 * Return the number of columns in an Armadillo umat.
 */
int mlpackNumColUmat(const char* identifier);

/**
 * Return the number of elements in an Armadillo umat.
 */
int mlpackNumElemUmat(const char* identifier);

/**
 * Return the number of elements in an Armadillo row.
 */
int mlpackNumElemRow(const char* identifier);

/**
 * Return the number of elements in an Armadillo urow.
 */
int mlpackNumElemUrow(const char* identifier);

/**
 * Return the number of elements in an Armadillo col.
 */
int mlpackNumElemCol(const char* identifier);

/**
 * Return the number of elements in an Armadillo ucol.
 */
int mlpackNumElemUcol(const char* identifier);

/**
 * Call CLI::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void mlpackToArmaMatWithInfo(const char* identifier,
                                    const bool* dimensions,
                                    double* memptr,
                                    const size_t rows,
                                    const size_t cols);

/**
 * Get the number of elements in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoElements(const char* identifier);

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoRows(const char* identifier);

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoCols(const char* identifier);

/**
 * Get a poconst size_t er to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
void* mlpackArmaPtrMatWithInfoPtr(const char* identifier);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
