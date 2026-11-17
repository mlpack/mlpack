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
  typename PointSelectionPolicy = KMeansSelection<>,
  // This parameter will be removed in mlpack 5.0.0!
  typename DeprecatedMatType = arma::mat
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
  [[deprecated("Will be removed in mlpack 5.0.0; use constructor that does not "
               "take data as a parameter instead, then pass the data to "
               "Apply().")]]
  NystroemMethod(const DeprecatedMatType& data,
                 KernelType& kernel,
                 const size_t rank);

  /**
   * Create the NystroemMethod object with the given kernel, rank, and point
   * selection policy.
   *
   * @param rank Rank to be used for matrix approximation.
   * @param kernel Kernel to be used for computation.
   * @param pointSelector Optional instantiated object to use for point
   *     selection.
   */
  NystroemMethod(const size_t rank = 100,
                 KernelType kernel = KernelType(),
                 PointSelectionPolicy pointSelector = PointSelectionPolicy());

  /**
   * Apply the low-rank factorization to obtain an output matrix G such that
   * K' = G * G^T.
   *
   * @param output Matrix to store kernel approximation into.
   */
  [[deprecated("Will be removed in mlpack 5.0.0; use other Apply() overload.")]]
  void Apply(DeprecatedMatType& output);

  /**
   * Apply the low-rank factorization to obtain an output matrix G such that
   * K' = G * G^T.
   *
   * @param data Matrix of data points.
   * @param output Matrix to store kernel approximation into.
   */
  template<typename MatType>
  void Apply(const MatType& data, MatType& output);

  /**
   * Construct the kernel matrix with matrix that contains the selected points.
   *
   * @param data Full data matrix.
   * @param selectedData Data matrix of points selected for the sample.
   * @param miniKernel to store the constructed mini-kernel matrix in.
   * @param semiKernel to store the constructed semi-kernel matrix in.
   */
  template<typename MatType>
  void GetKernelMatrix(const MatType& data,
                       const MatType& selectedData,
                       MatType& miniKernel,
                       MatType& semiKernel);

  /**
   * Construct the kernel matrix with the selected points.
   *
   * @param data Full data matrix.
   * @param selectedPoints Indices of selected points.
   * @param miniKernel to store the constructed mini-kernel matrix in.
   * @param semiKernel to store the constructed semi-kernel matrix in.
   */
  template<typename MatType>
  void GetKernelMatrix(const MatType& data,
                       const arma::Col<size_t>& selectedPoints,
                       MatType& miniKernel,
                       MatType& semiKernel);

  // Get the number of samples used for approximation.
  size_t Samples() const { return rank; }
  // Set the number of samples to use for approximation.
  void Samples(const size_t newSamples) { rank = newSamples; }

  // Get the kernel object.
  const KernelType& Kernel() const { return kernel; }
  // Modify the kernel object.
  KernelType& Kernel() { return kernel; }

  // Get the point selection policy.
  const PointSelectionPolicy& PointSelector() const { return pointSelector; }
  // Modify the point selection policy.
  PointSelectionPolicy& PointSelector() { return pointSelector; }

  // Serialize the object.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  // The reference dataset.
  const DeprecatedMatType* deprecatedData;
  // The locally stored kernel, if it is necessary.
  KernelType kernel;
  // Instantiated policy for point selection.
  PointSelectionPolicy pointSelector;
  // Rank used for matrix approximation.
  size_t rank;
};

} // namespace mlpack

// Include implementation.
#include "nystroem_method_impl.hpp"

#endif
