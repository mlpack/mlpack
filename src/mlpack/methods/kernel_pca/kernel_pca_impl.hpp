/**
 * @file kernel_pca_impl.hpp
 * @author Ajinkya Kale
 * @author Marcus Edel
 *
 * Implementation of Kernel PCA class to perform Kernel Principal Components
 * Analysis on the specified data set.
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
#ifndef __MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_IMPL_HPP
#define __MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_IMPL_HPP

// In case it hasn't already been included.
#include "kernel_pca.hpp"

namespace mlpack {
namespace kpca {

template <typename KernelType, typename KernelRule>
KernelPCA<KernelType, KernelRule>::KernelPCA(const KernelType kernel,
                                 const bool centerTransformedData) :
      kernel(kernel),
      centerTransformedData(centerTransformedData)
{ }

//! Apply Kernel Principal Component Analysis to the provided data set.
template <typename KernelType, typename KernelRule>
void KernelPCA<KernelType, KernelRule>::Apply(const arma::mat& data,
                                  arma::mat& transformedData,
                                  arma::vec& eigval,
                                  arma::mat& eigvec,
                                  const size_t newDimension)
{
  KernelRule::ApplyKernelMatrix(data, transformedData, eigval,
                                eigvec, newDimension, kernel);

  // Center the transformed data, if the user asked for it.
  if (centerTransformedData)
  {
    arma::colvec transformedDataMean = arma::mean(transformedData, 1);
    transformedData = transformedData - (transformedDataMean *
        arma::ones<arma::rowvec>(transformedData.n_cols));
  }
}

//! Apply Kernel Principal Component Analysis to the provided data set.
template <typename KernelType, typename KernelRule>
void KernelPCA<KernelType, KernelRule>::Apply(const arma::mat& data,
                                  arma::mat& transformedData,
                                  arma::vec& eigval,
                                  arma::mat& eigvec)
{
  Apply(data, transformedData, eigval, eigvec, data.n_cols);
}

//! Apply Kernel Principal Component Analysis to the provided data set.
template <typename KernelType, typename KernelRule>
void KernelPCA<KernelType, KernelRule>::Apply(const arma::mat& data,
                                  arma::mat& transformedData,
                                  arma::vec& eigVal)
{
  arma::mat coeffs;
  Apply(data, transformedData, eigVal, coeffs, data.n_cols);
}

//! Use KPCA for dimensionality reduction.
template <typename KernelType, typename KernelRule>
void KernelPCA<KernelType, KernelRule>::Apply(arma::mat& data,
                                    const size_t newDimension)
{
  arma::mat coeffs;
  arma::vec eigVal;

  Apply(data, data, eigVal, coeffs, newDimension);

  if (newDimension < coeffs.n_rows && newDimension > 0)
    data.shed_rows(newDimension, data.n_rows - 1);
}

//! Returns a string representation of the object.
template <typename KernelType, typename KernelRule>
std::string KernelPCA<KernelType, KernelRule>::ToString() const
{
  std::ostringstream convert;
  convert << "KernelPCA [" << this << "]" << std::endl;
  convert << "  Center Transformed: " << centerTransformedData <<std::endl;
  convert << "  Kernel Type: " << std::endl;
  convert <<  mlpack::util::Indent(kernel.ToString(),2);
  convert << std::endl;
  return convert.str();
}

}; // namespace mlpack
}; // namespace kpca

#endif
