/**
 * @file pca.hpp
 * @author Ajinkya Kale
 *
 * Defines the PCA class to perform Principal Components Analysis on the
 * specified data set.
 */
#ifndef __MLPACK_METHODS_PCA_PCA_HPP
#define __MLPACK_METHODS_PCA_PCA_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace pca {

class PCA
{
 public:
  PCA(const bool centerData = false, const bool scaleData = false);

  /**
   * Apply Principal Component Analysis to the provided data set.
   *
   * @param data - Data matrix
   * @param transformedData - Data with PCA applied
   * @param eigVal - contains eigen values in a column vector
   * @param coeff - PCA Loadings/Coeffs/EigenVectors
   */
  void Apply(const arma::mat& data, arma::mat& transformedData, arma::vec&
             eigVal, arma::mat& coeff) const;

  /**
   * Apply Principal Component Analysis to the provided data set.
   *
   * @param data - Data matrix
   * @param transformedData - Data with PCA applied
   * @param eigVal - contains eigen values in a column vector
   */
  void Apply(const arma::mat& data, arma::mat& transformedData,
             arma::vec& eigVal) const;

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
  void Apply(arma::mat& data, const size_t newDimension) const;

  /**
   * Delete PCA object
   */
  ~PCA();

  //! Get whether or not this PCA object will center the data.
  bool CenterData() const { return centerData; }
  //! Modify whether or not this PCA object will center the data when PCA is
  //! is performed.
  bool& CenterData() {  return centerData; }

  //! Get whether or not this PCA object will scale (by standard deviation) the
  //! data when PCA is performed.
  bool ScaleData() const { return scaleData; }
  //! Modify whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool& ScaleData() { return scaleData; }

 private:
  bool centerData;
  bool scaleData;
}; // class PCA

}; // namespace pca
}; // namespace mlpack

#endif
