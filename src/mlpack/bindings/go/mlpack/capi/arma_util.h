#ifndef MLPACK_BINDINGS_GO_MLPACK_ARMA_UTIL_H
#define MLPACK_BINDINGS_GO_MLPACK_ARMA_UTIL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void MLPACK_ToArma_mat(const char *identifier, const double mat[], int row, int col);

extern void MLPACK_ToArma_row(const char *identifier, const double rowvec[], int elem);

extern void MLPACK_ToArma_col(const char *identifier, const double colvec[], int elem);

extern void *MLPACK_ArmaPtr_mat(const char *identifier);

extern void *MLPACK_ArmaPtr_umat(const char *identifier);

extern void *MLPACK_ArmaPtr_row(const char *identifier);

extern void *MLPACK_ArmaPtr_urow(const char *identifier);

extern void *MLPACK_ArmaPtr_col(const char *identifier);

extern void *MLPACK_ArmaPtr_ucol(const char *identifier);

extern int MLPACK_NumRow_mat(const char *identifier);

extern int MLPACK_NumCol_mat(const char *identifier);

extern int MLPACK_NumElem_mat(const char *identifier);

extern int MLPACK_NumRow_umat(const char *identifier);

extern int MLPACK_NumCol_umat(const char *identifier);

extern int MLPACK_NumElem_umat(const char *identifier);

extern int MLPACK_Size_row(const char *identifier);

extern int MLPACK_Size_urow(const char *identifier);

extern int MLPACK_NumElem_row(const char *identifier);

extern int MLPACK_NumElem_urow(const char *identifier);

extern int MLPACK_Size_col(const char *identifier);

extern int MLPACK_Size_ucol(const char *identifier);

extern int MLPACK_NumElem_col(const char *identifier);

extern int MLPACK_NumElem_ucol(const char *identifier);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
