/**
 * @file kde_rules.hpp
 * @author Roberto Hueso (robertohueso96@gmail.com)
 *
 * Rules Kernel Density estimation, so that it can be done with arbitrary tree
 * types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_KDE_RULES_HPP
#define MLPACK_METHODS_KDE_RULES_HPP

#include <mlpack/core/tree/traversal_info.hpp>

namespace mlpack {
namespace kde {

template<typename MetricType, typename KernelType, typename TreeType>
class KDERules
{
 public:
  KDERules(const arma::mat& referenceSet,
           const arma::mat& querySet,
           arma::vec& densities,
           const double relError,
           const double absError,
           const std::vector<size_t>& oldFromNewQueries,
           MetricType& metric,
           KernelType& kernel);

  //! Base Case
  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  //! SingleTree Rescore
  double Score(const size_t queryIndex, TreeType& referenceNode);

  //! SingleTree Score
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore) const;

  //! DoubleTree Score
  double Score(TreeType& queryNode, TreeType& referenceNode);

  //! DoubleTree Rescore
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore) const;

  typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;

  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }

  TraversalInfoType& TraversalInfo() { return traversalInfo; }

  //! Get the number of base cases.
  size_t BaseCases() const { return baseCases; }

  //! Get the number of scores.
  size_t Scores() const { return scores; }

 private:
  //! The reference set.
  const arma::mat& referenceSet;

  //! The query set.
  const arma::mat& querySet;

  //! Density values.
  arma::vec& densities;

  //! Absolute error tolerance.
  const double absError;

  //! Relatve error tolerance.
  const double relError;

  //! New query dataset order.
  const std::vector<size_t>& oldFromNewQueries;

  //! Instantiated metric.
  MetricType& metric;

  //! Instantiated kernel
  KernelType& kernel;

  //! The last query index.
  size_t lastQueryIndex;

  //! The last reference index.
  size_t lastReferenceIndex;

  TraversalInfoType traversalInfo;

  //! The number of base cases.
  size_t baseCases;

  //! The number of scores.
  size_t scores;

  // Check TreeType is supported.
  static_assert(!tree::TreeTraits<TreeType>::HasDuplicatedPoints,
                "TreeType must not have duplicated points.");
  static_assert(tree::TreeTraits<TreeType>::UniqueNumDescendants,
                "TreeType must provide a number of unique descendants.");
};

} // namespace kde
} // namespace mlpack

// Include implementation.
#include "kde_rules_impl.hpp"

#endif
