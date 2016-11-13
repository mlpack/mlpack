/**
 * @file kernel_pca.hpp
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
#include <mlpack/methods/kernel_pca/kernel_rules/naive_method.hpp>

namespace mlpack {
namespace kpca {

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
  typename KernelRule = NaiveKernelRule<KernelType>
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
            const bool centerTransformedData = false);

  /**
   * Apply Kernel Principal Components Analysis to the provided data set.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to output results into.
   * @param eigval KPCA eigenvalues will be written to this vector.
   * @param eigvec KPCA eigenvectors will be written to this matrix.
   * @param newDimension New dimension for the dataset.
   */
  void Apply(const arma::mat& data,
             arma::mat& transformedData,
             arma::vec& eigval,
             arma::mat& eigvec,
             const size_t newDimension);

  /**
   * Apply Kernel Principal Components Analysis to the provided data set.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to output results into.
   * @param eigval KPCA eigenvalues will be written to this vector.
   * @param eigvec KPCA eigenvectors will be written to this matrix.
   */
  void Apply(const arma::mat& data,
             arma::mat& transformedData,
             arma::vec& eigval,
             arma::mat& eigvec);

  /**
   * Apply Kernel Principal Component Analysis to the provided data set.
   *
   * @param data Data matrix.
   * @param transformedData Matrix to output results into.
   * @param eigval KPCA eigenvalues will be written to this vector.
   */
  void Apply(const arma::mat& data,
             arma::mat& transformedData,
             arma::vec& eigval);

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
  void Apply(arma::mat& data, const size_t newDimension);

  //! Get the kernel.
  const KernelType& Kernel() const { return kernel; }
  //! Modify the kernel.
  KernelType& Kernel() { return kernel; }

  //! Return whether or not the transformed data is centered.
  bool CenterTransformedData() const { return centerTransformedData; }
  //! Return whether or not the transformed data is centered.
  bool& CenterTransformedData() { return centerTransformedData; }

 private:
  //! The instantiated kernel.
  KernelType kernel;
  //! If true, the data will be scaled (by standard deviation) when Apply() is
  //! run.
  bool centerTransformedData;

}; // class KernelPCA

} // namespace kpca
} // namespace mlpack

// Include implementation.
#include "kernel_pca_impl.hpp"

#endif // MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP
