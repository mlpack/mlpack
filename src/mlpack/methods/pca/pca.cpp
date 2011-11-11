/**
 * @file pca.cpp
 *
 * Implementation of PCA class to perform Principal Components Analysis on the
 * specified data set.
 */
#include "pca.hpp"
#include <mlpack/core.h>

namespace mlpack {
namespace pca {

PCA::PCA()
{
}

/**
 * Apply Armadillo's Principal Component Analysis on the data set.
 */
void PCA::Apply(const arma::mat& data, arma::mat& coeff, arma::mat& score)
{
  //Armadillo's PCA api
  arma::princomp(coeff, score, arma::trans(data));
}

PCA::~PCA()
{
}

}; // namespace mlpack
}; // namespace pca
