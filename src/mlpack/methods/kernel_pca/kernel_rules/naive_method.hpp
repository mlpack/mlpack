/**
 * @file naive_method.hpp
 * @author Ajinkya Kale
 *
 * Use the naive method to construct the kernel matrix.
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

#ifndef __MLPACK_METHODS_KERNEL_PCA_NAIVE_METHOD_HPP
#define __MLPACK_METHODS_KERNEL_PCA_NAIVE_METHOD_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kpca {

template<typename KernelType>
class NaiveKernelRule
{
 public:
  public:
    /**
     * Construct the kernel matrix approximation using the nystroem method.
     *
     * @param data Input data points.
     * @param transformedData Matrix to output results into.
     * @param eigval KPCA eigenvalues will be written to this vector.
     * @param eigvec KPCA eigenvectors will be written to this matrix.
     * @param rank Rank to be used for matrix approximation.
     * @param kernel Kernel to be used for computation.
     */
    static void ApplyKernelMatrix(const arma::mat& data,
                                  arma::mat& transformedData,
                                  arma::vec& eigval,
                                  arma::mat& eigvec,
                                  const size_t /* unused */,
                                  KernelType kernel = KernelType())
  {
    // Construct the kernel matrix.
    arma::mat kernelMatrix;
    // Resize the kernel matrix to the right size.
    kernelMatrix.set_size(data.n_cols, data.n_cols);

    // Note that we only need to calculate the upper triangular part of the 
    // kernel matrix, since it is symmetric. This helps minimize the number of
    // kernel evaluations.
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      for (size_t j = i; j < data.n_cols; ++j)
      {
        // Evaluate the kernel on these two points.
        kernelMatrix(i, j) = kernel.Evaluate(data.unsafe_col(i),
                                             data.unsafe_col(j));
      }
    }

    // Copy to the lower triangular part of the matrix.
    for (size_t i = 1; i < data.n_cols; ++i)
      for (size_t j = 0; j < i; ++j)
        kernelMatrix(i, j) = kernelMatrix(j, i);

    // For PCA the data has to be centered, even if the data is centered. But it
    // is not guaranteed that the data, when mapped to the kernel space, is also
    // centered. Since we actually never work in the feature space we cannot
    // center the data. So, we perform a "psuedo-centering" using the kernel
    // matrix.
    arma::rowvec rowMean = arma::sum(kernelMatrix, 0) / kernelMatrix.n_cols;
    kernelMatrix.each_col() -= arma::sum(kernelMatrix, 1) / kernelMatrix.n_cols;
    kernelMatrix.each_row() -= rowMean;
    kernelMatrix += arma::sum(rowMean) / kernelMatrix.n_cols;

    // Eigendecompose the centered kernel matrix.
    arma::eig_sym(eigval, eigvec, kernelMatrix);

    // Swap the eigenvalues since they are ordered backwards (we need largest to
    // smallest).
    for (size_t i = 0; i < floor(eigval.n_elem / 2.0); ++i)
      eigval.swap_rows(i, (eigval.n_elem - 1) - i);

    // Flip the coefficients to produce the same effect.
    eigvec = arma::fliplr(eigvec);

    transformedData = eigvec.t() * kernelMatrix;
  }
};

}; // namespace kpca
}; // namespace mlpack

#endif
