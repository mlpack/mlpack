/**
 * @file methods/pca/pca.hpp
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

#include "decomposition_policies/decomposition_policies.hpp"

namespace mlpack {

/**
 * This class implements principal components analysis (PCA). This is a
 * common, widely-used technique that is often used for either dimensionality
 * reduction or transforming data into a better basis.  Further information on
 * PCA can be found in almost any statistics or machine learning textbook, and
 * all over the internet.
 */
template<typename DecompositionPolicy = ExactSVDPolicy>
class PCA
{
 public:
  /**
   * Create the PCA object, specifying if the data should be scaled in each
   * dimension by standard deviation when PCA is performed.
   *
   * @param scaleData Whether or not to scale the data.
   * @param decomposition Decomposition policy to use.
   */
  PCA(const bool scaleData = false,
      const DecompositionPolicy& decomposition = DecompositionPolicy());

  /**
   * Apply Principal Component Analysis to the provided data set. It is safe
   * to pass the same matrix reference for both data and transformedData.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to put results of PCA into.
   * @param eigVal Vector to put eigenvalues into.
   * @param eigvec Matrix to put eigenvectors (loadings) into.
   */
  template<typename MatType = arma::mat,
           typename OutMatType = arma::mat,
           typename VecType = arma::vec>
  void Apply(const MatType& data,
             OutMatType& transformedData,
             VecType& eigVal,
             OutMatType& eigvec);

  /**
   * Apply Principal Component Analysis to the provided data set. It is safe
   * to pass the same matrix reference for both data and transformedData.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to store results of PCA in.
   * @param eigVal Vector to put eigenvalues into.
   */
  template<typename MatType = arma::mat,
           typename OutMatType = arma::mat,
           typename VecType = arma::vec>
  void Apply(const MatType& data,
             OutMatType& transformedData,
             VecType& eigVal);
  /**
   * Apply Principal Component Analysis to the provided data set. It is safe
   * to pass the same matrix reference for both data and transformedData.
   * @param data Data matrix.
   * @param transformedData Matrix to store results of PCA in.
   */
  template<typename MatType = arma::mat, typename OutMatType = arma::mat>
  void Apply(const MatType& data,
             OutMatType& transformedData);

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
  template<typename MatType = arma::mat>
  double Apply(MatType& data, const size_t newDimension);

  //! This overload is here to make sure int gets casted right to size_t.
  template<typename MatType = arma::mat>
  inline double Apply(MatType& data, const int newDimension)
  {
    return Apply(data, size_t(newDimension));
  }

  /**
   * Use PCA for dimensionality reduction on the given dataset. This will save
   * the newDimension largest principal components of the data and remove the
   * rest, storing the result in `transformedData`. The return value is the
   * amount of variance of the data that is retained; this is a value between 0
   * and 1.  For instance, a value of 0.9 indicates that 90% of the variance
   * present in the data was retained.
   *
   * @param data Data matrix.
   * @param transformedData Output matrix to store transformed data in.
   * @param newDimension New dimension of the data.
   * @return Amount of the variance of the data retained (between 0 and 1).
   */
  template<typename MatType = arma::mat, typename OutMatType = arma::mat>
  double Apply(const MatType& data,
               OutMatType& transformedData,
               const size_t newDimension);

  //! This overload is here to make sure int gets casted right to size_t.
  template<typename MatType = arma::mat, typename OutMatType = arma::mat>
  inline double Apply(const MatType& data,
                      OutMatType& transformedData,
                      const int newDimension)
  {
    return Apply(data, transformedData, size_t(newDimension));
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
  template<typename MatType>
  double Apply(MatType& data, const double varRetained);

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
  template<typename MatType = arma::mat, typename OutMatType = arma::mat>
  double Apply(const MatType& data,
               OutMatType& transformedData,
               const double varRetained);

  //! Get whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool ScaleData() const { return scaleData; }
  //! Modify whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool& ScaleData() { return scaleData; }

 private:
  //! Scaling the data is when we reduce the variance of each dimension to 1.
  template<typename MatType>
  void ScaleData(MatType& centeredData)
  {
    if (scaleData)
    {
      // Scaling the data is when we reduce the variance of each dimension
      // to 1. We do this by dividing each dimension by its standard
      // deviation.
      arma::Col<typename MatType::elem_type> stdDev = arma::stddev(
          centeredData, 0, 1 /* for each dimension */);

      // If there are any zeroes, make them very small.
      for (size_t i = 0; i < stdDev.n_elem; ++i)
        if (stdDev[i] == 0)
          stdDev[i] = 1e-50;

      centeredData /= repmat(stdDev, 1, centeredData.n_cols);
    }
  }

  //! Whether or not the data will be scaled by standard deviation when PCA is
  //! performed.
  bool scaleData;

  //! Decomposition method used to perform principal components analysis.
  DecompositionPolicy decomposition;
}; // class PCA

} // namespace mlpack

// Include implementation.
#include "pca_impl.hpp"

#endif
