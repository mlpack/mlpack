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

#include "kde_stat.hpp"

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
  typedef TreeType<MetricType, kde::KDEStat, MatType> Tree;

  KDE();

  KDE(const double bandwidth,
      const double relError = 1e-5,
      const double absError = 0,
      const bool breadthFirst = false);

  KDE(MetricType& metric,
      KernelType& kernel,
      const double relError = 1e-5,
      const double absError = 0,
      const bool breadthFirst = false);

  KDE(const KDE& other);

  KDE(KDE&& other);

  KDE& operator=(KDE other);

  ~KDE();

  void Train(const MatType& referenceSet);

  void Train(Tree& referenceTree);

  void Evaluate(const MatType& querySet, arma::vec& estimations);

  void Evaluate(Tree& queryTree,
                const std::vector<size_t>& oldFromNewQueries,
                arma::vec& estimations);

  const KernelType& Kernel() const { return kernel; }

  KernelType& Kernel() { return kernel; }

  const Tree& ReferenceTree() const { return referenceTree; }

  //! Get relative error tolerance.
  double RelativeError() const { return relError; }

  //! Modify relative error tolerance.
  void RelativeError(const double newError);

  //! Get absolute error tolerance.
  double AbsoluteError() const { return absError; }

  //! Modify absolute error tolerance.
  void AbsoluteError(const double newError);

  //! Get whether breadth-first traversal is being used.
  bool BreadthFirst() const { return breadthFirst; }

  //! Modify whether breadth-first traversal is being used.
  bool& BreadthFirst() { return breadthFirst; }

  //! Check if reference tree is owned by the KDE model.
  bool OwnsReferenceTree() const { return ownsReferenceTree; }

  //! Check if KDE model is trained or not.
  bool IsTrained() const { return trained; }

 private:
  KernelType* kernel;

  MetricType* metric;

  Tree* referenceTree;

  double relError;

  double absError;

  bool breadthFirst;

  bool ownsKernel;

  bool ownsMetric;

  bool ownsReferenceTree;

  bool trained;
};

} // namespace kde
} // namespace mlpack

// Include implementation.
#include "kde_impl.hpp"

#endif // MLPACK_METHODS_KDE_KDE_HPP
