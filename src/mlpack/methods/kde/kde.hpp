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

  KDE(const double bandwidth = 1.0,
      const double relError = 1e-5,
      const double absError = 0,
      const bool breadthFirst = false);

  ~KDE();

  void Train(const MatType& referenceSet);

  void Train(Tree& referenceTree);

  void Evaluate(const MatType& querySet, arma::vec& estimations);

  void Evaluate(Tree& queryTree,
                const std::vector<size_t>& oldFromNewQueries,
                arma::vec& estimations);

 private:
  KernelType* kernel;

  Tree* referenceTree;

  double relError;

  double absError;

  bool breadthFirst;

  bool ownsReferenceTree;

  bool trained;
};

} // namespace kde
} // namespace mlpack

// Include implementation.
#include "kde_impl.hpp"

#endif // MLPACK_METHODS_KDE_KDE_HPP
