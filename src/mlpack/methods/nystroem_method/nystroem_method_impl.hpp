/**
 * @file nystroem_method_impl.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Nystroem method for approximating a kernel matrix.
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_IMPL_HPP
#define __MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_IMPL_HPP

// In case it hasn't been included yet.
#include "nystroem_method.hpp"

namespace mlpack {
namespace kernel {

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
    for (size_t j = 0; j < rank; ++j)
      miniKernel(i, j) = kernel.Evaluate(data.col(selectedPoints(i)),
                                         data.col(selectedPoints(j)));

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

  // Construct the output matrix.
  arma::mat normalization = arma::diagmat(1.0 / sqrt(s));
  output = semiKernel * U * normalization * V;
}

}; // namespace kernel
}; // namespace mlpack

#endif
