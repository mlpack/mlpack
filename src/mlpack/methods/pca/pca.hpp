/**
 * @file pca.hpp
 * @author Ajinkya Kale
 *
 * Defines the PCA class to perform Principal Components Analysis on the
 * specified data set.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
   * Apply Principal Component Analysis to the provided data set.  It is safe to
   * pass the same matrix reference for both data and transformedData.
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
   * Apply Principal Component Analysis to the provided data set.  It is safe to
   * pass the same matrix reference for both data and transformedData.
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
   * rest.  The parameter returned is the amount of variance of the data that is
   * retained; this is a value between 0 and 1.  For instance, a value of 0.9
   * indicates that 90% of the variance present in the data was retained.
   *
   * @param data Data matrix.
   * @param newDimension New dimension of the data.
   * @return Amount of the variance of the data retained (between 0 and 1).
   */
  double Apply(arma::mat& data, const size_t newDimension) const;

  //! This overload is here to make sure int gets casted right to size_t.
  inline double Apply(arma::mat& data, const int newDimension) const
  {
    return Apply(data, size_t(newDimension));
  }

  /**
   * Use PCA for dimensionality reduction on the given dataset.  This will save
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
  double Apply(arma::mat& data, const double varRetained) const;

  //! Get whether or not this PCA object will scale (by standard deviation) the
  //! data when PCA is performed.
  bool ScaleData() const { return scaleData; }
  //! Modify whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool& ScaleData() { return scaleData; }

  // Returns a string representation of this object. 
  std::string ToString() const;

 private:
  //! Whether or not the data will be scaled by standard deviation when PCA is
  //! performed.
  bool scaleData;

}; // class PCA

}; // namespace pca
}; // namespace mlpack

#endif
