/**
 * @file methods/nystroem_method/nystroem_method.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Nystroem method for approximating a kernel matrix.
 * There are many variations on how to do this, so template parameters allow the
 * selection of many different techniques.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_HPP
#define MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_HPP

#include <mlpack/core.hpp>

#include "kmeans_selection.hpp"
#include "ordered_selection.hpp"
#include "random_selection.hpp"

namespace mlpack {

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
   * @param semiKernel to store the constructed semi-kernel matrix in.
   */
  void GetKernelMatrix(const arma::mat* data,
                       arma::mat& miniKernel,
                       arma::mat& semiKernel);

  /**
   * Construct the kernel matrix with the selected points.
   *
   * @param selectedPoints Indices of selected points.
   * @param miniKernel to store the constructed mini-kernel matrix in.
   * @param semiKernel to store the constructed semi-kernel matrix in.
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

} // namespace mlpack

// Include implementation.
#include "nystroem_method_impl.hpp"

#endif
