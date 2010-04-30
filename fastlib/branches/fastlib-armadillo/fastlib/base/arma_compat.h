/***
 * @file arma_compat.h
 *
 * Compatibility layer for GenMatrix and arma::mat classes.
 * This is not written to be fast but to provide an intermediate compatibility
 * layer while converting from GenMatrix to arma::matrix internally.
 *
 */

#ifndef ARMA_COMPAT_H
#define ARMA_COMPAT_H

#include <armadillo>
#include "../la/matrix.h"

namespace arma_compat {

  void armaToMatrix(const arma::mat &mat, Matrix &mg);
  void matrixToArma(const Matrix &gm, arma::mat &mat);

  // write a column into an uninitialized vector
  void armaColVector(const arma::mat &mat, int col, Vector &v);
}

#endif /* ARMA_COMPAT_H */
