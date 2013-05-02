/**
 * @file fastmks_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of FastMKSRules for cover tree search.
 *
 * This file is part of MLPACK 1.0.4.
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
    kernel(kernel)
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
}

template<typename KernelType, typename TreeType>
inline force_inline
double FastMKSRules<KernelType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{

  double kernelEval = kernel.Evaluate(querySet.unsafe_col(queryIndex),
                                      referenceSet.unsafe_col(referenceIndex));

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

template<typename MetricType, typename TreeType>
double FastMKSRules<MetricType, TreeType>::Score(const size_t queryIndex,
                                                 TreeType& referenceNode) const
{
  // Calculate the maximum possible kernel value.
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const arma::vec refCentroid;
  referenceNode.Bound().Centroid(refCentroid);

  const double maxKernel = kernel.Evaluate(queryPoint, refCentroid) +
      referenceNode.FurthestDescendantDistance() * queryKernels[queryIndex];

  // Compare with the current best.
  const double bestKernel = products(products.n_rows - 1, queryIndex);

  // We return the inverse of the maximum kernel so that larger kernels are
  // recursed into first.
  return (maxKernel > bestKernel) ? (1.0 / maxKernel) : DBL_MAX;
}

template<typename MetricType, typename TreeType>
double FastMKSRules<MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double baseCaseResult) const
{
  // We already have the base case result.  Add the bound.
  const double maxKernel = baseCaseResult +
      referenceNode.FurthestDescendantDistance() * queryKernels[queryIndex];
  const double bestKernel = products(products.n_rows - 1, queryIndex);

  // We return the inverse of the maximum kernel so that larger kernels are
  // recursed into first.
  return (maxKernel > bestKernel) ? (1.0 / maxKernel) : DBL_MAX;
}

template<typename MetricType, typename TreeType>
double FastMKSRules<MetricType, TreeType>::Score(TreeType& queryNode,
                                                 TreeType& referenceNode) const
{
  // Calculate the maximum possible kernel value.
  const arma::vec queryCentroid;
  const arma::vec refCentroid;
  queryNode.Bound().Centroid(queryCentroid);
  referenceNode.Bound().Centroid(refCentroid);

  const double refKernelTerm = queryNode.FurthestDescendantDistance() *
      referenceNode.Stat().SelfKernel();
  const double queryKernelTerm = referenceNode.FurthestDescendantDistance() *
      queryNode.Stat().SelfKernel();

  const double maxKernel = kernel.Evaluate(queryCentroid, refCentroid) +
      refKernelTerm + queryKernelTerm +
      (queryNode.FurthestDescendantDistance() *
       referenceNode.FurthestDescendantDistance());

  // The existing bound.
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestKernel = queryNode.Stat().Bound();

  // We return the inverse of the maximum kernel so that larger kernels are
  // recursed into first.
  return (maxKernel > bestKernel) ? (1.0 / maxKernel) : DBL_MAX;
}

template<typename MetricType, typename TreeType>
double FastMKSRules<MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double baseCaseResult) const
{
  // We already have the base case, so we need to add the bounds.
  const double refKernelTerm = queryNode.FurthestDescendantDistance() *
      referenceNode.Stat().SelfKernel();
  const double queryKernelTerm = referenceNode.FurthestDescendantDistance() *
      queryNode.Stat().SelfKernel();

  const double maxKernel = baseCaseResult + refKernelTerm + queryKernelTerm +
      (queryNode.FurthestDescendantDistance() *
       referenceNode.FurthestDescendantDistance());

  // The existing bound.
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestKernel = queryNode.Stat().Bound();

  // We return the inverse of the maximum kernel so that larger kernels are
  // recursed into first.
  return (maxKernel > bestKernel) ? (1.0 / maxKernel) : DBL_MAX;
}

template<typename MetricType, typename TreeType>
double FastMKSRules<MetricType, TreeType>::Rescore(const size_t queryIndex,
                                                   TreeType& /*referenceNode*/,
                                                   const double oldScore) const
{
  const double bestKernel = products(products.n_rows - 1, queryIndex);

  return ((1.0 / oldScore) > bestKernel) ? oldScore : DBL_MAX;
}

template<typename MetricType, typename TreeType>
double FastMKSRules<MetricType, TreeType>::Rescore(TreeType& queryNode,
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
template<typename MetricType, typename TreeType>
double FastMKSRules<MetricType, TreeType>::CalculateBound(TreeType& queryNode)
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
//  double bestPointSelfKernel = -DBL_MAX;
  const double queryDescendantDistance = queryNode.FurthestDescendantDistance();

  // Loop over all points in this node to find the best and worst.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const size_t point = queryNode.Point(i);
    if (products(products.n_rows - 1, point) < worstPointKernel)
      worstPointKernel = products(products.n_rows - 1, point);

    if (products(products.n_rows - 1, point) == -DBL_MAX)
      continue; // Avoid underflow.

    const double candidateKernel = products(products.n_rows - 1, point) -
        (2 * queryDescendantDistance) *
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
template<typename MetricType, typename TreeType>
void FastMKSRules<MetricType, TreeType>::InsertNeighbor(const size_t queryIndex,
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
