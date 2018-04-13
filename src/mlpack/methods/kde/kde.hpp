/**
 * @file kde.hpp
 * @author Roberto Hueso (robertohueso96@gmail.com)
 *
 * Kernel Density Estimation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_KDE_KDE_HPP
#define MLPACK_METHODS_KDE_KDE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>

namespace mlpack {
namespace kde /** Kernel Density Estimation. */ {

template<typename MetricType = mlpack::metric::EuclideanDistance,
         typename MatType = arma::mat,
         typename KernelType = kernel::GaussianKernel,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = tree::KDTree>
class KDE
{
 public:
  typedef TreeType<MetricType, tree::EmptyStatistic, MatType> Tree;

  KDE(const MatType& referenceSet,
      const double error = 1e-8,
      const double bandwidth = 1.0,
      const size_t leafSize = 2);

  ~KDE();

  void Evaluate(const MatType& query, arma::vec& estimations);

 private:
  const MatType& referenceSet;

  KernelType* kernel;

  Tree* referenceTree;

  double error;

  double bandwidth;

  const int leafSize;
};

} // namespace kde
} // namespace mlpack

// Include implementation.
#include "kde_impl.hpp"

#endif // MLPACK_METHODS_KDE_KDE_HPP
