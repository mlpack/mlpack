/**
 * @file fastmks_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of FastMKSRules for cover tree search.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_FASTMKS_FASTMKS_RULES_IMPL_HPP
#define MLPACK_METHODS_FASTMKS_FASTMKS_RULES_IMPL_HPP

// In case it hasn't already been included.
#include "fastmks_rules.hpp"

namespace mlpack {
namespace fastmks {

template<typename KernelType, typename TreeType>
FastMKSRules<KernelType, TreeType>::FastMKSRules(
    const typename TreeType::Mat& referenceSet,
    const typename TreeType::Mat& querySet,
    const size_t k,
    KernelType& kernel) :
    referenceSet(referenceSet),
    querySet(querySet),
    k(k),
    kernel(kernel),
    lastQueryIndex(-1),
    lastReferenceIndex(-1),
    lastKernel(0.0),
    baseCases(0),
    scores(0)
{
  // Precompute each self-kernel.
  queryKernels.set_size(querySet.n_cols);
  for (size_t i = 0; i < querySet.n_cols; ++i)
    queryKernels[i] = sqrt(kernel.Evaluate(querySet.col(i),
                                           querySet.col(i)));

  referenceKernels.set_size(referenceSet.n_cols);
  for (size_t i = 0; i < referenceSet.n_cols; ++i)
    referenceKernels[i] = sqrt(kernel.Evaluate(referenceSet.col(i),
                                               referenceSet.col(i)));

  // Set to invalid memory, so that the first node combination does not try to
  // dereference null pointers.
  traversalInfo.LastQueryNode() = (TreeType*) this;
  traversalInfo.LastReferenceNode() = (TreeType*) this;

  // Let's build the list of candidate points for each query point.
  // It will be initialized with k candidates: (-DBL_MAX, size_t() - 1)
  // The list of candidates will be updated when visiting new points with the
  // BaseCase() method.
  const Candidate def = std::make_pair(-DBL_MAX, size_t() - 1);

  CandidateList pqueue;
  pqueue.reserve(k);
  for (size_t i = 0; i < k; i++)
    pqueue.push(def);
  std::vector<CandidateList> tmp(querySet.n_cols, pqueue);
  candidates.swap(tmp);
}

template<typename KernelType, typename TreeType>
void FastMKSRules<KernelType, TreeType>::GetResults(
    arma::Mat<size_t>& indices,
    arma::mat& products)
{
  indices.set_size(k, querySet.n_cols);
  products.set_size(k, querySet.n_cols);

  for (size_t i = 0; i < querySet.n_cols; i++)
  {
    CandidateList& pqueue = candidates[i];
    for (size_t j = 1; j <= k; j++)
    {
      indices(k - j, i) = pqueue.top().second;
      products(k - j, i) = pqueue.top().first;
      pqueue.pop();
    }
  }
}

template<typename KernelType, typename TreeType>
inline force_inline
double FastMKSRules<KernelType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // Score() always happens before BaseCase() for a given node combination.  For
  // cover trees, the kernel evaluation between the two centroid points already
  // happened.  So we don't need to do it.  Note that this optimizes out if the
  // first conditional is false (its result is known at compile time).
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    if ((queryIndex == lastQueryIndex) &&
        (referenceIndex == lastReferenceIndex))
      return lastKernel;

    // Store new values.
    lastQueryIndex = queryIndex;
    lastReferenceIndex = referenceIndex;
  }

  ++baseCases;
  double kernelEval = kernel.Evaluate(querySet.col(queryIndex),
                                      referenceSet.col(referenceIndex));

  // Update the last kernel value, if we need to.
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
    lastKernel = kernelEval;

  // If the reference and query sets are identical, we still need to compute the
  // base case (so that things can be bounded properly), but we won't add it to
  // the results.
  if ((&querySet == &referenceSet) && (queryIndex == referenceIndex))
    return kernelEval;

  InsertNeighbor(queryIndex, referenceIndex, kernelEval);

  return kernelEval;
}

template<typename KernelType, typename TreeType>
double FastMKSRules<KernelType, TreeType>::Score(const size_t queryIndex,
                                                 TreeType& referenceNode)
{
  // Compare with the current best.
  const double bestKernel = candidates[queryIndex].top().first;

  // See if we can perform a parent-child prune.
  const double furthestDist = referenceNode.FurthestDescendantDistance();
  if (referenceNode.Parent() != NULL)
  {
    double maxKernelBound;
    const double parentDist = referenceNode.ParentDistance();
    const double combinedDistBound = parentDist + furthestDist;
    const double lastKernel = referenceNode.Parent()->Stat().LastKernel();
    if (kernel::KernelTraits<KernelType>::IsNormalized)
    {
      const double squaredDist = std::pow(combinedDistBound, 2.0);
      const double delta = (1 - 0.5 * squaredDist);
      if (lastKernel <= delta)
      {
        const double gamma = combinedDistBound * sqrt(1 - 0.25 * squaredDist);
        maxKernelBound = lastKernel * delta +
             gamma * sqrt(1 - std::pow(lastKernel, 2.0));
      }
      else
      {
        maxKernelBound = 1.0;
      }
    }
    else
    {
      maxKernelBound = lastKernel +
          combinedDistBound * queryKernels[queryIndex];
    }

    if (maxKernelBound < bestKernel)
      return DBL_MAX;
  }

  // Calculate the maximum possible kernel value, either by calculating the
  // centroid or, if the centroid is a point, use that.
  ++scores;
  double kernelEval;
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    // Could it be that this kernel evaluation has already been calculated?
    if (tree::TreeTraits<TreeType>::HasSelfChildren &&
        referenceNode.Parent() != NULL &&
        referenceNode.Point(0) == referenceNode.Parent()->Point(0))
    {
      kernelEval = referenceNode.Parent()->Stat().LastKernel();
    }
    else
    {
      kernelEval = BaseCase(queryIndex, referenceNode.Point(0));
    }
  }
  else
  {
    arma::vec refCenter;
    referenceNode.Center(refCenter);

    kernelEval = kernel.Evaluate(querySet.col(queryIndex), refCenter);
  }

  referenceNode.Stat().LastKernel() = kernelEval;

  double maxKernel;
  if (kernel::KernelTraits<KernelType>::IsNormalized)
  {
    const double squaredDist = std::pow(furthestDist, 2.0);
    const double delta = (1 - 0.5 * squaredDist);
    if (kernelEval <= delta)
    {
      const double gamma = furthestDist * sqrt(1 - 0.25 * squaredDist);
      maxKernel = kernelEval * delta +
          gamma * sqrt(1 - std::pow(kernelEval, 2.0));
    }
    else
    {
      maxKernel = 1.0;
    }
  }
  else
  {
    maxKernel = kernelEval + furthestDist * queryKernels[queryIndex];
  }

  // We return the inverse of the maximum kernel so that larger kernels are
  // recursed into first.
  return (maxKernel >= bestKernel) ? (1.0 / maxKernel) : DBL_MAX;
}

template<typename KernelType, typename TreeType>
double FastMKSRules<KernelType, TreeType>::Score(TreeType& queryNode,
                                                 TreeType& referenceNode)
{
  // Update and get the query node's bound.
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestKernel = queryNode.Stat().Bound();

  // First, see if we can make a parent-child or parent-parent prune.  These
  // four bounds on the maximum kernel value are looser than the bound normally
  // used, but they can prevent a base case from needing to be calculated.

  // Convenience caching so lines are shorter.
  const double queryParentDist = queryNode.ParentDistance();
  const double queryDescDist = queryNode.FurthestDescendantDistance();
  const double refParentDist = referenceNode.ParentDistance();
  const double refDescDist = referenceNode.FurthestDescendantDistance();
  double adjustedScore = traversalInfo.LastBaseCase();

  const double queryDistBound = (queryParentDist + queryDescDist);
  const double refDistBound = (refParentDist + refDescDist);
  double dualQueryTerm;
  double dualRefTerm;

  // The parent-child and parent-parent prunes work by applying the same pruning
  // condition as when the parent node was used, except they are tighter because
  //    queryDistBound < queryNode.Parent()->FurthestDescendantDistance()
  // and
  //    refDistBound < referenceNode.Parent()->FurthestDescendantDistance()
  // so we construct the same bounds that were used when Score() was called with
  // the parents, except with the tighter distance bounds.  Sometimes this
  // allows us to prune nodes without evaluating the base cases between them.
  if (traversalInfo.LastQueryNode() == queryNode.Parent())
  {
    // We can assume that queryNode.Parent() != NULL, because at the root node
    // combination, the traversalInfo.LastQueryNode() pointer will _not_ be
    // NULL.  We also should be guaranteed that
    // traversalInfo.LastReferenceNode() is either the reference node or the
    // parent of the reference node.
    adjustedScore += queryDistBound *
        traversalInfo.LastReferenceNode()->Stat().SelfKernel();
    dualQueryTerm = queryDistBound;
  }
  else
  {
    // The query parent could be NULL, which does weird things and we have to
    // consider.
    if (traversalInfo.LastReferenceNode() != NULL)
    {
      adjustedScore += queryDescDist *
          traversalInfo.LastReferenceNode()->Stat().SelfKernel();
      dualQueryTerm = queryDescDist;
    }
    else
    {
      // This makes it so a child-parent (or parent-parent) prune is not
      // possible.
      dualQueryTerm = 0.0;
      adjustedScore = bestKernel;
    }
  }

  if (traversalInfo.LastReferenceNode() == referenceNode.Parent())
  {
    // We can assume that referenceNode.Parent() != NULL, because at the root
    // node combination, the traversalInfo.LastReferenceNode() pointer will
    // _not_ be NULL.
    adjustedScore += refDistBound *
        traversalInfo.LastQueryNode()->Stat().SelfKernel();
    dualRefTerm = refDistBound;
  }
  else
  {
    // The reference parent could be NULL, which does weird things and we have
    // to consider.
    if (traversalInfo.LastQueryNode() != NULL)
    {
      adjustedScore += refDescDist *
          traversalInfo.LastQueryNode()->Stat().SelfKernel();
      dualRefTerm = refDescDist;
    }
    else
    {
      // This makes it so a child-parent (or parent-parent) prune is not
      // possible.
      dualRefTerm = 0.0;
      adjustedScore = bestKernel;
    }
  }

  // Now add the dual term.
  adjustedScore += (dualQueryTerm * dualRefTerm);

  if (adjustedScore < bestKernel)
  {
    // It is not possible that this node combination can contain a point
    // combination with kernel value better than the minimum kernel value to
    // improve any of the results, so we can prune it.
    return DBL_MAX;
  }

  // We were unable to perform a parent-child or parent-parent prune, so now we
  // must calculate kernel evaluation, if necessary.
  double kernelEval = 0.0;
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    // For this type of tree, we may have already calculated the base case in
    // the parents.
    if ((traversalInfo.LastQueryNode() != NULL) &&
        (traversalInfo.LastReferenceNode() != NULL) &&
        (traversalInfo.LastQueryNode()->Point(0) == queryNode.Point(0)) &&
        (traversalInfo.LastReferenceNode()->Point(0) == referenceNode.Point(0)))
    {
      // Base case already done.
      kernelEval = traversalInfo.LastBaseCase();

      // When BaseCase() is called after Score(), these must be correct so that
      // another kernel evaluation is not performed.
      lastQueryIndex = queryNode.Point(0);
      lastReferenceIndex = referenceNode.Point(0);
    }
    else
    {
      // The kernel must be evaluated, but it is between points in the dataset,
      // so we can call BaseCase().  BaseCase() will set lastQueryIndex and
      // lastReferenceIndex correctly.
      kernelEval = BaseCase(queryNode.Point(0), referenceNode.Point(0));
    }

    traversalInfo.LastBaseCase() = kernelEval;
  }
  else
  {
    // Calculate the maximum possible kernel value.
    arma::vec queryCenter;
    arma::vec refCenter;
    queryNode.Center(queryCenter);
    referenceNode.Center(refCenter);

    kernelEval = kernel.Evaluate(queryCenter, refCenter);

    traversalInfo.LastBaseCase() = kernelEval;
  }
  ++scores;

  double maxKernel;
  if (kernel::KernelTraits<KernelType>::IsNormalized)
  {
    // We have a tighter bound for normalized kernels.
    const double querySqDist = std::pow(queryDescDist, 2.0);
    const double refSqDist = std::pow(refDescDist, 2.0);
    const double bothSqDist = std::pow((queryDescDist + refDescDist), 2.0);

    if (kernelEval <= (1 - 0.5 * bothSqDist))
    {
      const double queryDelta = (1 - 0.5 * querySqDist);
      const double queryGamma = queryDescDist * sqrt(1 - 0.25 * querySqDist);
      const double refDelta = (1 - 0.5 * refSqDist);
      const double refGamma = refDescDist * sqrt(1 - 0.25 * refSqDist);

      maxKernel = kernelEval * (queryDelta * refDelta - queryGamma * refGamma) +
          sqrt(1 - std::pow(kernelEval, 2.0)) *
          (queryGamma * refDelta + queryDelta * refGamma);
    }
    else
    {
      maxKernel = 1.0;
    }
  }
  else
  {
    // Use standard bound; kernel is not normalized.
    const double refKernelTerm = queryDescDist *
        referenceNode.Stat().SelfKernel();
    const double queryKernelTerm = refDescDist * queryNode.Stat().SelfKernel();

    maxKernel = kernelEval + refKernelTerm + queryKernelTerm +
        (queryDescDist * refDescDist);
  }

  // Store relevant information for parent-child pruning.
  traversalInfo.LastQueryNode() = &queryNode;
  traversalInfo.LastReferenceNode() = &referenceNode;

  // We return the inverse of the maximum kernel so that larger kernels are
  // recursed into first.
  return (maxKernel >= bestKernel) ? (1.0 / maxKernel) : DBL_MAX;
}

template<typename KernelType, typename TreeType>
double FastMKSRules<KernelType, TreeType>::Rescore(const size_t queryIndex,
                                                   TreeType& /*referenceNode*/,
                                                   const double oldScore) const
{
  const double bestKernel = candidates[queryIndex].top().first;

  return ((1.0 / oldScore) >= bestKernel) ? oldScore : DBL_MAX;
}

template<typename KernelType, typename TreeType>
double FastMKSRules<KernelType, TreeType>::Rescore(TreeType& queryNode,
                                                   TreeType& /*referenceNode*/,
                                                   const double oldScore) const
{
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestKernel = queryNode.Stat().Bound();

  return ((1.0 / oldScore) >= bestKernel) ? oldScore : DBL_MAX;
}

/**
 * Calculate the bound for the given query node.  This bound represents the
 * minimum value which a node combination must achieve to guarantee an
 * improvement in the results.
 *
 * @param queryNode Query node to calculate bound for.
 */
template<typename KernelType, typename TreeType>
double FastMKSRules<KernelType, TreeType>::CalculateBound(TreeType& queryNode)
    const
{
  // We have four possible bounds -- just like NeighborSearchRules, but they are
  // slightly different in this context.
  //
  // (1) min ( min_{all points p in queryNode} P_p[k],
  //           min_{all children c in queryNode} B(c) );
  // (2) max_{all points p in queryNode} P_p[k] + (worst child distance + worst
  //           descendant distance) sqrt(K(I_p[k], I_p[k]));
  // (3) max_{all children c in queryNode} B(c) + <-- not done yet.  ignored.
  // (4) B(parent of queryNode);
  double worstPointKernel = DBL_MAX;
  double bestAdjustedPointKernel = -DBL_MAX;

  const double queryDescendantDistance = queryNode.FurthestDescendantDistance();

  // Loop over all points in this node to find the worst max-kernel value and
  // the best possible adjusted max-kernel value that could be held by any
  // descendant.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const size_t point = queryNode.Point(i);
    const CandidateList& candidatesPoints = candidates[point];
    if (candidatesPoints.top().first < worstPointKernel)
      worstPointKernel = candidatesPoints.top().first;

    if (candidatesPoints.top().first == -DBL_MAX)
      continue; // Avoid underflow.

    // This should be (queryDescendantDistance + centroidDistance) for any tree
    // but it works for cover trees since centroidDistance = 0 for cover trees.
    // The formulation here is slightly different than in Equation 43 of
    // "Dual-tree fast exact max-kernel search".  Because we could be searching
    // for k max kernels and not just one, the bound for this point must
    // actually be the minimum adjusted kernel of all k candidate kernels.
    // So,
    //   B(N_q) = min_{1 \le j \le k} k_j^*(p_q) -
    //       \lambda_q \sqrt(K(p_j^*(p_q), p_j^*(p_q)))
    // where p_j^*(p_q) is the j'th kernel candidate for query point p_q and
    // k_j^*(p_q) is K(p_q, p_j^*(p_q)).
    double worstPointCandidateKernel = DBL_MAX;
    typedef typename CandidateList::const_iterator iter;
    for (iter it = candidatesPoints.begin(); it != candidatesPoints.end(); ++it)
    {
      const double candidateKernel = it->first - queryDescendantDistance *
          referenceKernels[it->second];
      if (candidateKernel < worstPointCandidateKernel)
        worstPointCandidateKernel = candidateKernel;
    }

    if (worstPointCandidateKernel > bestAdjustedPointKernel)
      bestAdjustedPointKernel = worstPointCandidateKernel;
  }

  // Loop over all the children in the node.
  double worstChildKernel = DBL_MAX;

  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    if (queryNode.Child(i).Stat().Bound() < worstChildKernel)
      worstChildKernel = queryNode.Child(i).Stat().Bound();
  }

  // Now assemble bound (1).
  const double firstBound = (worstPointKernel < worstChildKernel) ?
      worstPointKernel : worstChildKernel;

  // Bound (2) is bestAdjustedPointKernel.
  const double fourthBound = (queryNode.Parent() == NULL) ? -DBL_MAX :
      queryNode.Parent()->Stat().Bound();

  // Pick the best of these bounds.
  const double interA = (firstBound > bestAdjustedPointKernel) ? firstBound :
      bestAdjustedPointKernel;
  const double interB = fourthBound;

  return (interA > interB) ? interA : interB;
}

/**
 * Helper function to insert a point into the list of candidate points.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param index Index of reference point which is being inserted.
 * @param product Kernel value for given candidate.
 */
template<typename KernelType, typename TreeType>
inline void FastMKSRules<KernelType, TreeType>::InsertNeighbor(
    const size_t queryIndex,
    const size_t index,
    const double product)
{
  CandidateList& pqueue = candidates[queryIndex];
  if (product > pqueue.top().first)
  {
    Candidate c = std::make_pair(product, index);
    pqueue.pop();
    pqueue.push(c);
  }
}

} // namespace fastmks
} // namespace mlpack

#endif
