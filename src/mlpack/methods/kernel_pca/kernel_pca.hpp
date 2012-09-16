/**
 * @file kernel_pca.hpp
 * @author Ajinkya Kale
 *
 * Defines the KernelPCA class to perform Kernel Principal Components Analysis
 * on the specified data set.
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
#ifndef __MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP
#define __MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

namespace mlpack {
namespace kpca {

template <typename KernelType>
class KernelPCA
{
 public:
  KernelPCA(const KernelType kernel = KernelType(),
            const bool scaleData = false);

  /**
   * Apply Kernel Principal Component Analysis to the provided data set.
   *
   * @param data - Data matrix
   * @param transformedData - Data with PCA applied
   * @param eigVal - contains eigen values in a column vector
   * @param coeff - PCA Loadings/Coeffs/EigenVectors
   */
  void Apply(const arma::mat& data,
             arma::mat& transformedData,
             arma::vec& eigVal,
             arma::mat& coeff);

  /**
   * Apply Kernel Principal Component Analysis to the provided data set.
   *
   * @param data - Data matrix
   * @param transformedData - Data with PCA applied
   * @param eigVal - contains eigen values in a column vector
   */
  void Apply(const arma::mat& data,
             arma::mat& transformedData,
             arma::vec& eigVal);

  /**
   * Apply Dimensionality Reduction using Kernel Principal Component Analysis
   * to the provided data set.
   *
   * @param data - M x N Data matrix
   * @param newDimension - matrix consisting of N column vectors,
   * where each vector is the projection of the corresponding data vector
   * from data matrix onto the basis vectors contained in the columns of
   * coeff/eigen vector matrix with only newDimension number of columns chosen.
   */
  void Apply(arma::mat& data, const size_t newDimension);

  //! Get the kernel.
  const KernelType& Kernel() const { return kernel; }
  //! Modify the kernel.
  KernelType& Kernel() { return kernel; }

  //! Return whether or not this KernelPCA object will scale (by standard
  //! deviation) the data when kernel PCA is performed.
  bool ScaleData() const { return scaleData; }
  //! Modify whether or not this KernelPCA object will scale (by standard
  //! deviation) the data when kernel PCA is performed.
  bool& ScaleData() { return scaleData; }

 private:
  //! The instantiated kernel.
  KernelType kernel;
  //! If true, the data will be scaled (by standard deviation) when Apply() is
  //! run.
  bool scaleData;

}; // class KernelPCA

}; // namespace kpca
}; // namespace mlpack

// Include implementation.
#include "kernel_pca_impl.hpp"

#endif // __MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP
