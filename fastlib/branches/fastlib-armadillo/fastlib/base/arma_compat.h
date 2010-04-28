/***
 * @file arma_compat.h
 *
 * Compatibility layer for GenMatrix and arma::mat classes.
 * This is not written to be fast but to provide an intermediate compatibility
 * layer while converting from GenMatrix to arma::matrix internally.
 *
 */

#include <armadillo>
#include "../la/matrix.h"

namespace arma_compat {

  void armaToMatrix(const arma::mat &mat, Matrix &mg);
  void matrixToArma(const Matrix &gm, arma::mat &mat);

}
