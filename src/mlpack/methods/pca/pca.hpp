/**
 * @file pca.hpp
 *
 * Defines the PCA class to perform Principal Components Analysis on the
 * specified data set.
 */
#include <mlpack/core.h>
namespace mlpack {
namespace pca {

class PCA
{
 public:
  PCA();

  /**
   * Apply Armadillo's Principal Component Analysis to the provided data set.
   *
   * @param data - Data matrix
   * @param coeff - PCA Loadings
   * @param score - contains the coordinates of the original data in the new coordinate system defined by the principal components
   */
  void Apply(const arma::mat& data, arma::mat& coeff, arma::mat& score);

  /**
   * Delete PCA object
   */
  ~PCA();

}; // class PCA

}; // namespace pca
}; // namespace mlpack
