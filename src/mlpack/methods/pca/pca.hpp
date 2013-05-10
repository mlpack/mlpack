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

/**
 * This class implements principal components analysis (PCA).  This is a common,
 * widely-used technique that is often used for either dimensionality reduction
 * or transforming data into a better basis.  Further information on PCA can be
 * found in almost any statistics or machine learning textbook, and all over the
 * internet.
 */
class PCA
{
 public:
  /**
   * Create the PCA object, specifying if the data should be scaled in each
   * dimension by standard deviation when PCA is performed.
   *
   * @param scaleData Whether or not to scale the data.
   */
  PCA(const bool scaleData = false);

  /**
   * Apply Principal Component Analysis to the provided data set.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to put results of PCA into.
   * @param eigval Vector to put eigenvalues into.
   * @param eigvec Matrix to put eigenvectors (loadings) into.
   */
  void Apply(const arma::mat& data,
             arma::mat& transformedData,
             arma::vec& eigval,
             arma::mat& eigvec) const;

  /**
   * Apply Principal Component Analysis to the provided data set.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to store results of PCA in.
   * @param eigval Vector to put eigenvalues into.
   */
  void Apply(const arma::mat& data,
             arma::mat& transformedData,
             arma::vec& eigVal) const;

  /**
   * Use PCA for dimensionality reduction on the given dataset.  This will save
   * the newDimension largest principal components of the data and remove the
   * rest.
   *
   * @param data Data matrix.
   * @param newDimension New dimension of the data.
   */
  void Apply(arma::mat& data, const size_t newDimension) const;

  //! Get whether or not this PCA object will scale (by standard deviation) the
  //! data when PCA is performed.
  bool ScaleData() const { return scaleData; }
  //! Modify whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool& ScaleData() { return scaleData; }

 private:
  //! Whether or not the data will be scaled by standard deviation when PCA is
  //! performed.
  bool scaleData;

}; // class PCA

}; // namespace pca
}; // namespace mlpack

#endif
