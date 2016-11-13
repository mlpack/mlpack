/**
 * @file pca.hpp
 * @author Ajinkya Kale
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Defines the PCA class to perform Principal Components Analysis on the
 * specified data set. There are many variations on how to do this, so
 * template parameters allow the selection of different techniques.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_PCA_PCA_HPP
#define MLPACK_METHODS_PCA_PCA_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>

namespace mlpack {
namespace pca {

/**
 * This class implements principal components analysis (PCA). This is a
 * common, widely-used technique that is often used for either dimensionality
 * reduction or transforming data into a better basis.  Further information on
 * PCA can be found in almost any statistics or machine learning textbook, and
 * all over the internet. Note this class will be changed to have the name PCA
 * in mlpack 3.0.0
 */
template<typename DecompositionPolicy = ExactSVDPolicy>
class PCAType
{
 public:
  /**
   * Create the PCA object, specifying if the data should be scaled in each
   * dimension by standard deviation when PCA is performed.
   *
   * @param scaleData Whether or not to scale the data.
   */
  PCAType(const bool scaleData = false,
          const DecompositionPolicy& decomposition = DecompositionPolicy());

  /**
   * Apply Principal Component Analysis to the provided data set. It is safe
   * to pass the same matrix reference for both data and transformedData.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to put results of PCA into.
   * @param eigval Vector to put eigenvalues into.
   * @param eigvec Matrix to put eigenvectors (loadings) into.
   */
  void Apply(const arma::mat& data,
             arma::mat& transformedData,
             arma::vec& eigVal,
             arma::mat& eigvec);

  /**
   * Apply Principal Component Analysis to the provided data set. It is safe
   * to pass the same matrix reference for both data and transformedData.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to store results of PCA in.
   * @param eigVal Vector to put eigenvalues into.
   */
  void Apply(const arma::mat& data,
             arma::mat& transformedData,
             arma::vec& eigVal);

  /**
   * Use PCA for dimensionality reduction on the given dataset. This will save
   * the newDimension largest principal components of the data and remove the
   * rest. The parameter returned is the amount of variance of the data that
   * is retained; this is a value between 0 and 1.  For instance, a value of
   * 0.9 indicates that 90% of the variance present in the data was retained.
   *
   * @param data Data matrix.
   * @param newDimension New dimension of the data.
   * @return Amount of the variance of the data retained (between 0 and 1).
   */
  double Apply(arma::mat& data, const size_t newDimension);

  //! This overload is here to make sure int gets casted right to size_t.
  inline double Apply(arma::mat& data, const int newDimension)
  {
    return Apply(data, size_t(newDimension));
  }

  /**
   * Use PCA for dimensionality reduction on the given dataset. This will save
   * as many dimensions as necessary to retain at least the given amount of
   * variance (specified by parameter varRetained).  The amount should be
   * between 0 and 1; if the amount is 0, then only 1 dimension will be
   * retained.  If the amount is 1, then all dimensions will be retained.
   *
   * The method returns the actual amount of variance retained, which will
   * always be greater than or equal to the varRetained parameter.
   *
   * @param data Data matrix.
   * @param varRetained Lower bound on amount of variance to retain; should be
   *     between 0 and 1.
   * @return Actual amount of variance retained (between 0 and 1).
   */
  double Apply(arma::mat& data, const double varRetained);

  //! Get whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool ScaleData() const { return scaleData; }
  //! Modify whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool& ScaleData() { return scaleData; }

 private:
  //! Scaling the data is when we reduce the variance of each dimension to 1.
  void ScaleData(arma::mat& centeredData)
  {
    if (scaleData)
    {
      // Scaling the data is when we reduce the variance of each dimension
      // to 1. We do this by dividing each dimension by its standard
      // deviation.
      arma::vec stdDev = arma::stddev(
          centeredData, 0, 1 /* for each dimension */);

      // If there are any zeroes, make them very small.
      for (size_t i = 0; i < stdDev.n_elem; ++i)
        if (stdDev[i] == 0)
          stdDev[i] = 1e-50;

      centeredData /= arma::repmat(stdDev, 1, centeredData.n_cols);
    }
  }

  //! Whether or not the data will be scaled by standard deviation when PCA is
  //! performed.
  bool scaleData;

  //! Decomposition method used to perform principal components analysis.
  DecompositionPolicy decomposition;
}; // class PCA

//! 3.0.0 TODO: break reverse-compatibility by changing PCAType to PCA.
typedef PCAType<ExactSVDPolicy> PCA;

} // namespace pca
} // namespace mlpack

// Include implementation.
#include "pca_impl.hpp"

#endif
