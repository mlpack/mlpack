/**
 * @file methods/kernel_pca/kernel_pca.hpp
 * @author Ajinkya Kale
 * @author Marcus Edel
 *
 * Defines the KernelPCA class to perform Kernel Principal Components Analysis
 * on the specified data set.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP
#define MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP

#include <mlpack/core.hpp>

#include "kernel_rules/naive_method.hpp"
#include "kernel_rules/nystroem_method.hpp"

namespace mlpack {

/**
 * This class performs kernel principal components analysis (Kernel PCA), for a
 * given kernel.  This is a standard machine learning technique and is
 * well-documented on the Internet and in standard texts.  It is often used as a
 * dimensionality reduction technique, and can also be useful in mapping
 * linearly inseparable classes of points to different spaces where they are
 * linearly separable.
 *
 * The performance of the method is highly dependent on the kernel choice.
 * There are numerous available kernels in the mlpack::kernel namespace (see
 * files in mlpack/core/kernels/) and it is easy to write your own; see other
 * implementations for examples.
 */
template <
  typename KernelType,
  typename KernelRule = NaiveKernelRule
>
class KernelPCA
{
 public:
  /**
   * Construct the KernelPCA object, optionally passing a kernel.  Optionally,
   * the transformed data can be centered about the origin; to do this, pass
   * 'true' for centerTransformedData.  This will take slightly longer (but not
   * much).
   *
   * @param kernel Kernel to be used for computation.
   * @param centerTransformedData Center transformed data.
   */
  KernelPCA(const KernelType kernel = KernelType(),
            const bool centerTransformedData = false,
            const KernelRule rule = KernelRule());

  /**
   * Apply Kernel Principal Components Analysis to the provided data set.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to output results into.
   * @param eigval KPCA eigenvalues will be written to this vector.
   * @param eigvec KPCA eigenvectors will be written to this matrix.
   * @param newDimension New dimension for the dataset.
   */
  template<typename MatType>
  void Apply(const MatType& data,
             MatType& transformedData,
             typename GetColType<MatType>::type& eigval,
             MatType& eigvec,
             const size_t newDimension);

  /**
   * Apply Kernel Principal Components Analysis to the provided data set.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to output results into.
   * @param eigval KPCA eigenvalues will be written to this vector.
   * @param eigvec KPCA eigenvectors will be written to this matrix.
   */
  template<typename MatType>
  void Apply(const MatType& data,
             MatType& transformedData,
             typename GetColType<MatType>::type& eigval,
             MatType& eigvec);

  /**
   * Apply Kernel Principal Component Analysis to the provided data set.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to output results into.
   * @param eigval KPCA eigenvalues will be written to this vector.
   */
  template<typename MatType>
  void Apply(const MatType& data,
             MatType& transformedData,
             typename GetColType<MatType>::type& eigval);

  /**
   * Apply Kernel Principal Component Analysis to the provided data set, keeping
   * only the top `newDimension` dimensions.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to output results into.
   * @param newDimensions Number of dimensions to keep.
   */
  template<typename MatType>
  void Apply(const MatType& data,
             MatType& transformedData,
             const size_t newDimension);

  /**
   * Apply dimensionality reduction using Kernel Principal Component Analysis
   * to the provided data set.  The data matrix will be modified in-place.  Note
   * that the dimension can be larger than the existing dimension because KPCA
   * works on the kernel matrix, not the covariance matrix.  This means the new
   * dimension can be as large as the number of points (columns) in the dataset.
   * Note that if you specify newDimension to be larger than the current
   * dimension of the data (the number of rows), then it's not really
   * "dimensionality reduction"...
   *
   * @param data Data matrix.
   * @param newDimension New dimension for the dataset.
   */
  template<typename MatType>
  [[deprecated("Will be removed in mlpack 5.0.0. "
               "Use the version with 'transformedData' instead.")]]
  void Apply(MatType& data, const size_t newDimension);

  // Get the kernel.
  const KernelType& Kernel() const { return kernel; }
  // Modify the kernel.
  KernelType& Kernel() { return kernel; }

  // Return whether or not the transformed data is centered.
  bool CenterTransformedData() const { return centerTransformedData; }
  // Return whether or not the transformed data is centered.
  bool& CenterTransformedData() { return centerTransformedData; }

  // Serialize the KernelPCA object.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  // The instantiated kernel.
  KernelType kernel;
  // If true, the data will be scaled (by standard deviation) when Apply() is
  // run.
  bool centerTransformedData;
  // The instantiated kernel matrix computation rule.
  KernelRule rule;
}; // class KernelPCA

} // namespace mlpack

// Include implementation.
#include "kernel_pca_impl.hpp"

#endif // MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP
