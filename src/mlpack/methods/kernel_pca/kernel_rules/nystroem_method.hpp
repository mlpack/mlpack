/**
 * @file nystroem_method.hpp
 * @author Marcus Edel
 *
 * Use the Nystroem method for approximating a kernel matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_KERNEL_PCA_NYSTROEM_METHOD_HPP
#define MLPACK_METHODS_KERNEL_PCA_NYSTROEM_METHOD_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/nystroem_method/kmeans_selection.hpp>
#include <mlpack/methods/nystroem_method/nystroem_method.hpp>

namespace mlpack {
namespace kpca {

template<
  typename KernelType,
  typename PointSelectionPolicy = kernel::KMeansSelection<>
>
class NystroemKernelRule
{
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
                                  const size_t rank,
                                  KernelType kernel = KernelType())
    {
      arma::mat G, v;
      kernel::NystroemMethod<KernelType, PointSelectionPolicy> nm(data, kernel,
                                                        rank);
      nm.Apply(G);
      transformedData = G.t() * G;

      // Center the reconstructed approximation.
      math::Center(transformedData, transformedData);

      // For PCA the data has to be centered, even if the data is centered. But
      // it is not guaranteed that the data, when mapped to the kernel space, is
      // also centered. Since we actually never work in the feature space we
      // cannot center the data. So, we perform a "psuedo-centering" using the
      // kernel matrix.
      arma::colvec colMean = arma::sum(G, 1) / G.n_rows;
      G.each_row() -= arma::sum(G, 0) / G.n_rows;
      G.each_col() -= colMean;
      G += arma::sum(colMean) / G.n_rows;

      // Eigendecompose the centered kernel matrix.
      arma::eig_sym(eigval, eigvec, transformedData);

      // Swap the eigenvalues since they are ordered backwards (we need largest
      // to smallest).
      for (size_t i = 0; i < floor(eigval.n_elem / 2.0); ++i)
        eigval.swap_rows(i, (eigval.n_elem - 1) - i);

      // Flip the coefficients to produce the same effect.
      eigvec = arma::fliplr(eigvec);

      transformedData = eigvec.t() * G.t();
    }
};

} // namespace kpca
} // namespace mlpack

#endif
