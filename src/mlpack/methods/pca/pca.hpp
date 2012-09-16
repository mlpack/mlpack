/**
 * @file pca.hpp
 * @author Ajinkya Kale
 *
 * Defines the PCA class to perform Principal Components Analysis on the
 * specified data set.
 *
 * This file is part of MLPACK 1.0.3.
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

class PCA
{
 public:
  PCA(const bool scaleData = false);

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

  //! Get whether or not this PCA object will scale (by standard deviation) the
  //! data when PCA is performed.
  bool ScaleData() const { return scaleData; }
  //! Modify whether or not this PCA object will scale (by standard deviation)
  //! the data when PCA is performed.
  bool& ScaleData() { return scaleData; }

 private:
  bool scaleData;
}; // class PCA

}; // namespace pca
}; // namespace mlpack

#endif
