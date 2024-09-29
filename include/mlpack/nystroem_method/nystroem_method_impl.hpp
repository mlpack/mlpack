/**
 * @file methods/nystroem_method/nystroem_method_impl.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Nystroem method for approximating a kernel matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_IMPL_HPP
#define MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_IMPL_HPP

// In case it hasn't been included yet.
#include "nystroem_method.hpp"

namespace mlpack {

template<typename KernelType, typename PointSelectionPolicy>
NystroemMethod<KernelType, PointSelectionPolicy>::NystroemMethod(
    const arma::mat& data,
    KernelType& kernel,
    const size_t rank) :
    data(data),
    kernel(kernel),
    rank(rank)
{ }

template<typename KernelType, typename PointSelectionPolicy>
void NystroemMethod<KernelType, PointSelectionPolicy>::GetKernelMatrix(
    const arma::mat* selectedData,
    arma::mat& miniKernel,
    arma::mat& semiKernel)
{
  // Assemble mini-kernel matrix.
  for (size_t i = 0; i < rank; ++i)
    for (size_t j = 0; j < rank; ++j)
      miniKernel(i, j) = kernel.Evaluate(selectedData->col(i),
                                         selectedData->col(j));

  // Construct semi-kernel matrix with interactions between selected data and
  // all points.
  for (size_t i = 0; i < data.n_cols; ++i)
    for (size_t j = 0; j < rank; ++j)
      semiKernel(i, j) = kernel.Evaluate(data.col(i),
                                         selectedData->col(j));
  // Clean the memory.
  delete selectedData;
}

template<typename KernelType, typename PointSelectionPolicy>
void NystroemMethod<KernelType, PointSelectionPolicy>::GetKernelMatrix(
    const arma::Col<size_t>& selectedPoints,
    arma::mat& miniKernel,
    arma::mat& semiKernel)
{
  // Assemble mini-kernel matrix.
  for (size_t i = 0; i < rank; ++i)
  {
    for (size_t j = 0; j < rank; ++j)
    {
      miniKernel(i, j) = kernel.Evaluate(data.col(selectedPoints(i)),
                                         data.col(selectedPoints(j)));
    }
  }

  // Construct semi-kernel matrix with interactions between selected points and
  // all points.
  for (size_t i = 0; i < data.n_cols; ++i)
    for (size_t j = 0; j < rank; ++j)
      semiKernel(i, j) = kernel.Evaluate(data.col(i),
                                         data.col(selectedPoints(j)));
}

template<typename KernelType, typename PointSelectionPolicy>
void NystroemMethod<KernelType, PointSelectionPolicy>::Apply(arma::mat& output)
{
  arma::mat miniKernel(rank, rank);
  arma::mat semiKernel(data.n_cols, rank);

  GetKernelMatrix(PointSelectionPolicy::Select(data, rank), miniKernel,
                  semiKernel);

  // Singular value decomposition mini-kernel matrix.
  arma::mat U, V;
  arma::vec s;
  arma::svd(U, s, V, miniKernel);

  // Construct the output matrix.  We need to have special handling when
  // miniKernel ended up being low-rank.
  arma::mat normalization = arma::diagmat(1.0 / sqrt(s));
  for (size_t i = 0; i < s.n_elem; ++i)
    if (std::abs(s[i]) <= 1e-20)
      normalization(i, i) = 0.0;

  output = semiKernel * U * normalization * V;
}

} // namespace mlpack

#endif
