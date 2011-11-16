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
   * Apply Principal Component Analysis to the provided data set.
   *
   * @param data - Data matrix
   * @param transformedData - Data with PCA applied
   * @param eigVal - contains eigen values in a column vector
   * @param coeff - PCA Loadings/Coeffs/EigenVectors
   */
  void Apply(const arma::mat& data, arma::mat& transformedData, arma::vec&
             eigVal, arma::mat& coeff);

  /**
   * Apply Principal Component Analysis to the provided data set.
   *
   * @param data - Data matrix
   * @param transformedData - Data with PCA applied
   * @param eigVal - contains eigen values in a column vector
   */
  void Apply(const arma::mat& data, arma::mat& transformedData,
             arma::vec& eigVal);

  /**
   * Apply Dimensionality Reduction using Principal Component Analysis
   * to the provided data set.
   *
   * @param data - M x N Data matrix
   * @param newDimension - matrix consisting of N column vectors,
   * where each vector is the projection of the corresponding data vector
   * from data matrix onto the basis vectors contained in the columns of
   * coeff/eigen vector matrix with only newDimension number of columns chosen.
   */
  void Apply(arma::mat& data, const int newDimension);

  /**
   * Delete PCA object
   */
  ~PCA();

}; // class PCA

}; // namespace pca
}; // namespace mlpack
