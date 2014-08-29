/**
 * @file fastmks_rules.hpp
 * @author Ryan Curtin
 *
 * Rules for the single or dual tree traversal for fast max-kernel search.
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
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree/cover_tree.hpp>

#include "../neighbor_search/ns_traversal_info.hpp"

namespace mlpack {
namespace fastmks {

/**
 * The base case and pruning rules for FastMKS (fast max-kernel search).
 */
template<typename KernelType, typename TreeType>
class FastMKSRules
{
 public:
  FastMKSRules(const arma::mat& referenceSet,
               const arma::mat& querySet,
               arma::Mat<size_t>& indices,
               arma::mat& products,
               KernelType& kernel);

  //! Compute the base case (kernel value) between two points.
  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate to be recursed into.
   */
  double Score(const size_t queryIndex, TreeType& referenceNode);

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * @param queryNode Candidate query node to be recursed into.
   * @param referenceNode Candidate reference node to be recursed into.
   */
  double Score(TreeType& queryNode, TreeType& referenceNode);

  /**
   * Re-evaluate the score for recursion order.  A low score indicates priority
   * for recursion, while DBL_MAX indicates that a node should not be recursed
   * into at all (it should be pruned).  This is used when the score has already
   * been calculated, but another recursion may have modified the bounds for
   * pruning.  So the old score is checked against the new pruning bound.
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   * @param oldScore Old score produced by Score() (or Rescore()).
   */
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore) const;

  /**
   * Re-evaluate the score for recursion order.  A low score indicates priority
   * for recursion, while DBL_MAX indicates that a node should not be recursed
   * into at all (it should be pruned).  This is used when the score has already
   * been calculated, but another recursion may have modified the bounds for
   * pruning.  So the old score is checked against the new pruning bound.
   *
   * @param queryNode Candidate query node to be recursed into.
   * @param referenceNode Candidate reference node to be recursed into.
   * @param oldScore Old score produced by Score() (or Rescore()).
   */
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore) const;

  //! Get the number of times BaseCase() was called.
  size_t BaseCases() const { return baseCases; }
  //! Modify the number of times BaseCase() was called.
  size_t& BaseCases() { return baseCases; }

  //! Get the number of times Score() was called.
  size_t Scores() const { return scores; }
  //! Modify the number of times Score() was called.
  size_t& Scores() { return scores; }

  typedef neighbor::NeighborSearchTraversalInfo<TreeType> TraversalInfoType;

  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  //! The reference dataset.
  const arma::mat& referenceSet;
  //! The query dataset.
  const arma::mat& querySet;

  //! The indices of the maximum kernel results.
  arma::Mat<size_t>& indices;
  //! The maximum kernels.
  arma::mat& products;

  //! Cached query set self-kernels (|| q || for each q).
  arma::vec queryKernels;
  //! Cached reference set self-kernels (|| r || for each r).
  arma::vec referenceKernels;

  //! The instantiated kernel.
  KernelType& kernel;

  //! The last query index BaseCase() was called with.
  size_t lastQueryIndex;
  //! The last reference index BaseCase() was called with.
  size_t lastReferenceIndex;
  //! The last kernel evaluation resulting from BaseCase().
  double lastKernel;

  //! Calculate the bound for a given query node.
  double CalculateBound(TreeType& queryNode) const;

  //! Utility function to insert neighbor into list of results.
  void InsertNeighbor(const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double distance);

  //! For benchmarking.
  size_t baseCases;
  //! For benchmarking.
  size_t scores;

  TraversalInfoType traversalInfo;
};

}; // namespace fastmks
}; // namespace mlpack

// Include implementation.
#include "fastmks_rules_impl.hpp"

#endif
