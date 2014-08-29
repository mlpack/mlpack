/**
 * @file fastmks_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of FastMKSRules for cover tree search.
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
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_IMPL_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_IMPL_HPP

// In case it hasn't already been included.
#include "fastmks_rules.hpp"

namespace mlpack {
namespace fastmks {

template<typename KernelType, typename TreeType>
FastMKSRules<KernelType, TreeType>::FastMKSRules(const arma::mat& referenceSet,
                                                 const arma::mat& querySet,
                                                 arma::Mat<size_t>& indices,
                                                 arma::mat& products,
                                                 KernelType& kernel) :
    referenceSet(referenceSet),
    querySet(querySet),
    indices(indices),
    products(products),
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
    queryKernels[i] = sqrt(kernel.Evaluate(querySet.unsafe_col(i),
                                           querySet.unsafe_col(i)));

  referenceKernels.set_size(referenceSet.n_cols);
  for (size_t i = 0; i < referenceSet.n_cols; ++i)
    referenceKernels[i] = sqrt(kernel.Evaluate(referenceSet.unsafe_col(i),
                                               referenceSet.unsafe_col(i)));

  // Set to invalid memory, so that the first node combination does not try to
  // dereference null pointers.
  traversalInfo.LastQueryNode() = (TreeType*) this;
  traversalInfo.LastReferenceNode() = (TreeType*) this;
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
  double kernelEval = kernel.Evaluate(querySet.unsafe_col(queryIndex),
                                      referenceSet.unsafe_col(referenceIndex));

  // Update the last kernel value, if we need to.
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
    lastKernel = kernelEval;

  // If the reference and query sets are identical, we still need to compute the
  // base case (so that things can be bounded properly), but we won't add it to
  // the results.
  if ((&querySet == &referenceSet) && (queryIndex == referenceIndex))
    return kernelEval;

  // If this is a better candidate, insert it into the list.
  if (kernelEval < products(products.n_rows - 1, queryIndex))
    return kernelEval;

  size_t insertPosition = 0;
  for ( ; insertPosition < products.n_rows; ++insertPosition)
    if (kernelEval >= products(insertPosition, queryIndex))
      break;

  InsertNeighbor(queryIndex, insertPosition, referenceIndex, kernelEval);

  return kernelEval;
}

template<typename KernelType, typename TreeType>
double FastMKSRules<KernelType, TreeType>::Score(const size_t queryIndex,
                                                 TreeType& referenceNode)
{
  // Compare with the current best.
  const double bestKernel = products(products.n_rows - 1, queryIndex);

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
    const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
    arma::vec refCentroid;
    referenceNode.Centroid(refCentroid);

    kernelEval = kernel.Evaluate(queryPoint, refCentroid);
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
  return (maxKernel > bestKernel) ? (1.0 / maxKernel) : DBL_MAX;
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
    arma::vec queryCentroid;
    arma::vec refCentroid;
    queryNode.Centroid(queryCentroid);
    referenceNode.Centroid(refCentroid);

    kernelEval = kernel.Evaluate(queryCentroid, refCentroid);

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
  return (maxKernel > bestKernel) ? (1.0 / maxKernel) : DBL_MAX;
}

template<typename KernelType, typename TreeType>
double FastMKSRules<KernelType, TreeType>::Rescore(const size_t queryIndex,
                                                   TreeType& /*referenceNode*/,
                                                   const double oldScore) const
{
  const double bestKernel = products(products.n_rows - 1, queryIndex);

  return ((1.0 / oldScore) > bestKernel) ? oldScore : DBL_MAX;
}

template<typename KernelType, typename TreeType>
double FastMKSRules<KernelType, TreeType>::Rescore(TreeType& queryNode,
                                                   TreeType& /*referenceNode*/,
                                                   const double oldScore) const
{
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestKernel = queryNode.Stat().Bound();

  return ((1.0 / oldScore) > bestKernel) ? oldScore : DBL_MAX;
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

  // Loop over all points in this node to find the best and worst.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const size_t point = queryNode.Point(i);
    if (products(products.n_rows - 1, point) < worstPointKernel)
      worstPointKernel = products(products.n_rows - 1, point);

    if (products(products.n_rows - 1, point) == -DBL_MAX)
      continue; // Avoid underflow.

    // This should be (queryDescendantDistance + centroidDistance) for any tree
    // but it works for cover trees since centroidDistance = 0 for cover trees.
    const double candidateKernel = products(products.n_rows - 1, point) -
        queryDescendantDistance *
        referenceKernels[indices(indices.n_rows - 1, point)];

    if (candidateKernel > bestAdjustedPointKernel)
      bestAdjustedPointKernel = candidateKernel;
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
//  const double interA = 0.0;
  const double interB = fourthBound;

  return (interA > interB) ? interA : interB;
}

/**
 * Helper function to insert a point into the neighbors and distances matrices.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param pos Position in list to insert into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename KernelType, typename TreeType>
void FastMKSRules<KernelType, TreeType>::InsertNeighbor(const size_t queryIndex,
                                                        const size_t pos,
                                                        const size_t neighbor,
                                                        const double distance)
{
  // We only memmove() if there is actually a need to shift something.
  if (pos < (products.n_rows - 1))
  {
    int len = (products.n_rows - 1) - pos;
    memmove(products.colptr(queryIndex) + (pos + 1),
        products.colptr(queryIndex) + pos,
        sizeof(double) * len);
    memmove(indices.colptr(queryIndex) + (pos + 1),
        indices.colptr(queryIndex) + pos,
        sizeof(size_t) * len);
  }

  // Now put the new information in the right index.
  products(pos, queryIndex) = distance;
  indices(pos, queryIndex) = neighbor;
}

}; // namespace fastmks
}; // namespace mlpack

#endif
