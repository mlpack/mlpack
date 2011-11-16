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
  void Apply(arma::mat& data, const int newDimension);
  void Apply(const arma::mat& data, arma::mat& transformedData,
             arma::vec& eigVal);
  void Apply(const arma::mat& data, arma::mat& transformedData, arma::vec&
             eigVal, arma::mat& coeffs);


  /*


  // And for someone who wants even more.
  ;*/

  /**
   * Delete PCA object
   */
  ~PCA();

}; // class PCA

}; // namespace pca
}; // namespace mlpack
