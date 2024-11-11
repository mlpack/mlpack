/**
 * @file methods/fastmks/fastmks_rules.hpp
 * @author Ryan Curtin
 *
 * Rules for the single or dual tree traversal for fast max-kernel search.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP
#define MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/tree/cover_tree/cover_tree.hpp>
#include <mlpack/core/tree/traversal_info.hpp>
#include <algorithm>

namespace mlpack {

/**
 * The FastMKSRules class is a template helper class used by FastMKS class when
 * performing exact max-kernel search. For each point in the query dataset, it
 * keeps track of the k best candidates in the reference dataset.
 *
 * @tparam KernelType Type of kernel to run FastMKS with.
 * @tparam TreeType Type of tree to run FastMKS with; it must satisfy the
 *     TreeType policy API.
 */
template<typename KernelType, typename TreeType>
class FastMKSRules
{
 public:
  /**
   * Construct the FastMKSRules object.  This is usually done from within the
   * FastMKS class at search time.
   *
   * @param referenceSet Set of reference data.
   * @param querySet Set of query data.
   * @param k Number of candidates to search for.
   * @param kernel Kernel to run FastMKS with.
   */
  FastMKSRules(const typename TreeType::Mat& referenceSet,
               const typename TreeType::Mat& querySet,
               const size_t k,
               KernelType& kernel);

  /**
   * Store the list of candidates for each query point in the given matrices.
   *
   * @param indices Matrix storing lists of candidate for each query point.
   * @param products Matrix storing kernel value for each candidate.
   */
  void GetResults(arma::Mat<size_t>& indices, arma::mat& products);

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

  using TraversalInfoType = mlpack::TraversalInfo<TreeType>;

  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

  //! Get the minimum number of base cases we need to perform to have acceptable
  //! results.
  size_t MinimumBaseCases() const { return k; }

 private:
  //! The reference dataset.
  const typename TreeType::Mat& referenceSet;
  //! The query dataset.
  const typename TreeType::Mat& querySet;

  //! Candidate represents a possible candidate point (value, index).
  using Candidate = std::pair<double, size_t>;

  //! Compare two candidates based on the value.
  struct CandidateCmp {
    bool operator()(const Candidate& c1, const Candidate& c2) const
    {
      return c1.first > c2.first;
    };
  };

  //! Set of candidates for each point.  We use a min-heap built on a
  //! std::vector to represent the list of candidate points for each query
  //! point.
  std::vector<std::vector<Candidate>> candidates;

  //! Number of points to search for.
  const size_t k;

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

  /**
   * Helper function to insert a point into the list of candidate points.
   *
   * @param queryIndex Index of point whose neighbors we are inserting into.
   * @param index Index of reference point which is being inserted.
   * @param product Kernel value for given candidate.
   */
  void InsertNeighbor(const size_t queryIndex,
                      const size_t index,
                      const double product);

  //! For benchmarking.
  size_t baseCases;
  //! For benchmarking.
  size_t scores;

  TraversalInfoType traversalInfo;
};

} // namespace mlpack

// Include implementation.
#include "fastmks_rules_impl.hpp"

#endif
