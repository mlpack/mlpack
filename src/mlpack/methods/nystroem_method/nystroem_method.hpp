/**
 * @file nystroem_method.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Nystroem method for approximating a kernel matrix.
 * There are many variations on how to do this, so template parameters allow the
 * selection of many different techniques.
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
#ifndef __MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_HPP
#define __MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_HPP

#include <mlpack/core.hpp>
#include "kmeans_selection.hpp"

namespace mlpack {
namespace kernel {

template<
  typename KernelType,
  typename PointSelectionPolicy = KMeansSelection<>
>
class NystroemMethod
{
 public:
  /**
   * Create the NystroemMethod object. The constructor here does not really do
   * anything.
   *
   * @param data Data matrix.
   * @param kernel Kernel to be used for computation.
   * @param rank Rank to be used for matrix approximation.
   */
  NystroemMethod(const arma::mat& data, KernelType& kernel, const size_t rank);

  /**
   * Apply the low-rank factorization to obtain an output matrix G such that
   * K' = G * G^T.
   *
   * @param output Matrix to store kernel approximation into.
   */
  void Apply(arma::mat& output);

  /**
   * Construct the kernel matrix with matrix that contains the selected points.
   *
   * @param data Data matrix pointer.
   * @param miniKernel to store the constructed mini-kernel matrix in.
   * @param miniKernel to store the constructed semi-kernel matrix in.
   */
  void GetKernelMatrix(const arma::mat* data, 
                       arma::mat& miniKernel, 
                       arma::mat& semiKernel);

  /**
   * Construct the kernel matrix with the selected points.
   *
   * @param points Indices of selected points.
   * @param miniKernel to store the constructed mini-kernel matrix in.
   * @param miniKernel to store the constructed semi-kernel matrix in.
   */
  void GetKernelMatrix(const arma::Col<size_t>& selectedPoints, 
                       arma::mat& miniKernel, 
                       arma::mat& semiKernel);

 private:
  //! The reference dataset.
  const arma::mat& data;
  //! The locally stored kernel, if it is necessary.
  KernelType& kernel;
  //! Rank used for matrix approximation.
  const size_t rank;
};

}; // namespace kernel
}; // namespace mlpack

// Include implementation.
#include "nystroem_method_impl.hpp"

#endif
