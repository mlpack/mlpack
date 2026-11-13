/**
 * @file methods/kernel_pca/kernel_pca_impl.hpp
 * @author Ajinkya Kale
 * @author Marcus Edel
 *
 * Implementation of Kernel PCA class to perform Kernel Principal Components
 * Analysis on the specified data set.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_IMPL_HPP
#define MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_IMPL_HPP

// In case it hasn't already been included.
#include "kernel_pca.hpp"

namespace mlpack {

template<typename KernelType, typename KernelRule>
KernelPCA<KernelType, KernelRule>::KernelPCA(
    const KernelType kernel,
    const bool centerTransformedData) :
    kernel(kernel),
    centerTransformedData(centerTransformedData)
{ }

// Apply Kernel Principal Component Analysis to the provided data set.
template<typename KernelType, typename KernelRule>
template<typename MatType>
void KernelPCA<KernelType, KernelRule>::Apply(
    const MatType& data,
    MatType& transformedData,
    typename GetColType<MatType>::type& eigval,
    MatType& eigvec,
    const size_t newDimension)
{
  KernelRule::ApplyKernelMatrix(data, transformedData, eigval,
                                eigvec, newDimension, kernel);

  // Center the transformed data, if the user asked for it.
  if (centerTransformedData)
  {
    typename GetColType<MatType>::type transformedDataMean =
        mean(transformedData, 1);
    transformedData = transformedData - (transformedDataMean *
        ones<typename GetRowType<MatType>::type>(transformedData.n_cols));
  }
}

// Apply Kernel Principal Component Analysis to the provided data set.
template<typename KernelType, typename KernelRule>
template<typename MatType>
void KernelPCA<KernelType, KernelRule>::Apply(
    const MatType& data,
    MatType& transformedData,
    typename GetColType<MatType>::type& eigval,
    MatType& eigvec)
{
  Apply(data, transformedData, eigval, eigvec, data.n_cols);
}

// Apply Kernel Principal Component Analysis to the provided data set.
template<typename KernelType, typename KernelRule>
template<typename MatType>
void KernelPCA<KernelType, KernelRule>::Apply(
    const MatType& data,
    MatType& transformedData,
    typename GetColType<MatType>::type& eigVal)
{
  MatType coeffs;
  Apply(data, transformedData, eigVal, coeffs, data.n_cols);
}

// Use KPCA for dimensionality reduction.
template<typename KernelType, typename KernelRule>
template<typename MatType>
void KernelPCA<KernelType, KernelRule>::Apply(
    const MatType& data,
    MatType& transformedData,
    const size_t newDimension)
{
  MatType coeffs;
  typename GetColType<MatType>::type eigVal;

  Apply(data, transformedData, eigVal, coeffs, newDimension);

  if (newDimension < coeffs.n_rows && newDimension > 0)
    transformedData.shed_rows(newDimension, data.n_rows - 1);
}

// Use KPCA for dimensionality reduction.
template<typename KernelType, typename KernelRule>
template<typename MatType>
void KernelPCA<KernelType, KernelRule>::Apply(
    MatType& data,
    const size_t newDimension)
{
  MatType coeffs;
  typename GetColType<MatType>::type eigVal;

  Apply(data, data, eigVal, coeffs, newDimension);

  if (newDimension < coeffs.n_rows && newDimension > 0)
    data.shed_rows(newDimension, data.n_rows - 1);
}

template<typename KernelType, typename KernelRule>
template<typename Archive>
void KernelPCA<KernelType, KernelRule>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar(CEREAL_NVP(kernel));
  ar(CEREAL_NVP(centerTransformedData));
}

} // namespace mlpack

#endif
