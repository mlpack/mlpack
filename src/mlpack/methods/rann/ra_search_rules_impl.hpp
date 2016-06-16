/**
 * @file ra_search_rules_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of RASearchRules.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_METHODS_RANN_RA_SEARCH_RULES_IMPL_HPP
#define MLPACK_METHODS_RANN_RA_SEARCH_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "ra_search_rules.hpp"

namespace mlpack {
namespace neighbor {

template<typename SortPolicy, typename MetricType, typename TreeType>
RASearchRules<SortPolicy, MetricType, TreeType>::
RASearchRules(const arma::mat& referenceSet,
              const arma::mat& querySet,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances,
              MetricType& metric,
              const double tau,
              const double alpha,
              const bool naive,
              const bool sampleAtLeaves,
              const bool firstLeafExact,
              const size_t singleSampleLimit,
              const bool sameSet) :
    referenceSet(referenceSet),
    querySet(querySet),
    neighbors(neighbors),
    distances(distances),
    metric(metric),
    sampleAtLeaves(sampleAtLeaves),
    firstLeafExact(firstLeafExact),
    singleSampleLimit(singleSampleLimit),
    sameSet(sameSet)
{
  // Validate tau to make sure that the rank approximation is greater than the
  // number of neighbors requested.

  // The rank approximation.
  const size_t n = referenceSet.n_cols;
  const size_t k = neighbors.n_rows;
  const size_t t = (size_t) std::ceil(tau * (double) n / 100.0);
  if (t < k)
  {
    Log::Warn << "Rank-approximation percentile " << tau << " corresponds to "
        << t << " points, which is less than k (" << k << ").";
    Log::Fatal << "Cannot return " << k << " approximate nearest neighbors "
        << "from the nearest " << t << " points.  Increase tau!" << std::endl;
  }
  else if (t == k)
    Log::Warn << "Rank-approximation percentile " << tau << " corresponds to "
        << t << " points; because k = " << k << ", this is exact search!"
        << std::endl;

  Timer::Start("computing_number_of_samples_reqd");
  numSamplesReqd = RAUtil::MinimumSamplesReqd(n, k, tau, alpha);
  Timer::Stop("computing_number_of_samples_reqd");

  // Initialize some statistics to be collected during the search.
  numSamplesMade = arma::zeros<arma::Col<size_t> >(querySet.n_cols);
  numDistComputations = 0;
  samplingRatio = (double) numSamplesReqd / (double) n;

  Log::Info << "Minimum samples required per query: " << numSamplesReqd <<
    ", sampling ratio: " << samplingRatio << std::endl;

  if (naive) // No tree traversal; just do naive sampling here.
  {
    // Sample enough points.
    for (size_t i = 0; i < querySet.n_cols; ++i)
    {
      arma::uvec distinctSamples;
      RAUtil::ObtainDistinctSamples(numSamplesReqd, n, distinctSamples);
      for (size_t j = 0; j < distinctSamples.n_elem; j++)
        BaseCase(i, (size_t) distinctSamples[j]);
    }
  }
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline force_inline
double RASearchRules<SortPolicy, MetricType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if (sameSet && (queryIndex == referenceIndex))
    return 0.0;

  double distance = metric.Evaluate(querySet.unsafe_col(queryIndex),
                                    referenceSet.unsafe_col(referenceIndex));

  // If this distance is better than any of the current candidates, the
  // SortDistance() function will give us the position to insert it into.
  arma::vec queryDist = distances.unsafe_col(queryIndex);
  arma::Col<size_t> queryIndices = neighbors.unsafe_col(queryIndex);
  size_t insertPosition = SortPolicy::SortDistance(queryDist, queryIndices,
      distance);

  // SortDistance() returns (size_t() - 1) if we shouldn't add it.
  if (insertPosition != (size_t() - 1))
    InsertNeighbor(queryIndex, insertPosition, referenceIndex, distance);

  numSamplesMade[queryIndex]++;

  // TO REMOVE
  numDistComputations++;

  return distance;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double distance = SortPolicy::BestPointToNodeDistance(queryPoint,
      &referenceNode);
  const double bestDistance = distances(distances.n_rows - 1, queryIndex);

  return Score(queryIndex, referenceNode, distance, bestDistance);
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double baseCaseResult)
{
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double distance = SortPolicy::BestPointToNodeDistance(queryPoint,
      &referenceNode, baseCaseResult);
  const double bestDistance = distances(distances.n_rows - 1, queryIndex);

  return Score(queryIndex, referenceNode, distance, bestDistance);
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double distance,
    const double bestDistance)
{
  // If this is better than the best distance we've seen so far, maybe there
  // will be something down this node.  Also check if enough samples are already
  // made for this query.
  if (SortPolicy::IsBetter(distance, bestDistance)
      && numSamplesMade[queryIndex] < numSamplesReqd)
  {
    // We cannot prune this node; try approximating it by sampling.

    // If we are required to visit the first leaf (to find possible duplicates),
    // make sure we do not approximate.
    if (numSamplesMade[queryIndex] > 0 || !firstLeafExact)
    {
      // Check if this node can be approximated by sampling.
      size_t samplesReqd = (size_t) std::ceil(samplingRatio *
          (double) referenceNode.NumDescendants());
      samplesReqd = std::min(samplesReqd,
          numSamplesReqd - numSamplesMade[queryIndex]);

      if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
      {
        // If too many samples required and not at a leaf, then can't prune.
        return distance;
      }
      else
      {
        if (!referenceNode.IsLeaf())
        {
          // Then samplesReqd <= singleSampleLimit.
          // Hence, approximate the node by sampling enough number of points.
          arma::uvec distinctSamples;
          RAUtil::ObtainDistinctSamples(samplesReqd,
              referenceNode.NumDescendants(),
                                distinctSamples);
          for (size_t i = 0; i < distinctSamples.n_elem; i++)
            // The counting of the samples are done in the 'BaseCase' function
            // so no book-keeping is required here.
            BaseCase(queryIndex, referenceNode.Descendant(distinctSamples[i]));

          // Node approximated, so we can prune it.
          return DBL_MAX;
        }
        else // We are at a leaf.
        {
          if (sampleAtLeaves) // If allowed to sample at leaves.
          {
            // Approximate node by sampling enough number of points.
            arma::uvec distinctSamples;
            RAUtil::ObtainDistinctSamples(samplesReqd,
                referenceNode.NumDescendants(),
                                  distinctSamples);
            for (size_t i = 0; i < distinctSamples.n_elem; i++)
              // The counting of the samples are done in the 'BaseCase' function
              // so no book-keeping is required here.
              BaseCase(queryIndex,
                  referenceNode.Descendant(distinctSamples[i]));

            // (Leaf) node approximated, so we can prune it.
            return DBL_MAX;
          }
          else
          {
            // Not allowed to sample from leaves, so cannot prune.
            return distance;
          }
        }
      }
    }
    else
    {
      // Try first to visit the first leaf to boost your accuracy and find
      // (near) duplicates if they exist.
      return distance;
    }
  }
  else
  {
    // Either there cannot be anything better in this node, or enough number of
    // samples are already made.  So prune it.

    // Add 'fake' samples from this node; they are fake because the distances to
    // these samples need not be computed.

    // If enough samples are already made, this step does not change the result
    // of the search.
    numSamplesMade[queryIndex] += (size_t) std::floor(
        samplingRatio * (double) referenceNode.NumDescendants());

    return DBL_MAX;
  }
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::
Rescore(const size_t queryIndex,
        TreeType& referenceNode,
        const double oldScore)
{
  // If we are already pruning, still prune.
  if (oldScore == DBL_MAX)
    return oldScore;

  // Just check the score again against the distances.
  const double bestDistance = distances(distances.n_rows - 1, queryIndex);

  // If this is better than the best distance we've seen so far,
  // maybe there will be something down this node.
  // Also check if enough samples are already made for this query.
  if (SortPolicy::IsBetter(oldScore, bestDistance)
      && numSamplesMade[queryIndex] < numSamplesReqd)
  {
    // We cannot prune this node; thus, we try approximating this node by
    // sampling.

    // Here, we assume that since we are re-scoring, the algorithm has already
    // sampled some candidates, and if specified, also traversed to the first
    // leaf.  So no check regarding that is made any more.

    // Check if this node can be approximated by sampling.
    size_t samplesReqd = (size_t) std::ceil(samplingRatio *
        (double) referenceNode.NumDescendants());
    samplesReqd = std::min(samplesReqd, numSamplesReqd -
        numSamplesMade[queryIndex]);

    if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
    {
      // If too many samples are required and we are not at a leaf, then we
      // can't prune.
      return oldScore;
    }
    else
    {
      if (!referenceNode.IsLeaf())
      {
        // Then, samplesReqd <= singleSampleLimit.  Hence, approximate the node
        // by sampling enough number of points.
        arma::uvec distinctSamples;
        RAUtil::ObtainDistinctSamples(samplesReqd,
            referenceNode.NumDescendants(), distinctSamples);
        for (size_t i = 0; i < distinctSamples.n_elem; i++)
          // The counting of the samples are done in the 'BaseCase' function so
          // no book-keeping is required here.
          BaseCase(queryIndex, referenceNode.Descendant(distinctSamples[i]));

        // Node approximated, so we can prune it.
        return DBL_MAX;
      }
      else // We are at a leaf.
      {
        if (sampleAtLeaves)
        {
          // Approximate node by sampling enough points.
          arma::uvec distinctSamples;
          RAUtil::ObtainDistinctSamples(samplesReqd,
              referenceNode.NumDescendants(), distinctSamples);
          for (size_t i = 0; i < distinctSamples.n_elem; i++)
            // The counting of the samples are done in the 'BaseCase' function
            // so no book-keeping is required here.
            BaseCase(queryIndex, referenceNode.Descendant(distinctSamples[i]));

          // (Leaf) node approximated, so we can prune it.
          return DBL_MAX;
        }
        else
        {
          // We cannot sample from leaves, so we cannot prune.
          return oldScore;
        }
      }
    }
  }
  else
  {
    // Either there cannot be anything better in this node, or enough number of
    // samples are already made, so prune it.

    // Add 'fake' samples from this node; they are fake because the distances to
    // these samples need not be computed.  If enough samples are already made,
    // this step does not change the result of the search.
    numSamplesMade[queryIndex] += (size_t) std::floor(samplingRatio *
        (double) referenceNode.NumDescendants());

    return DBL_MAX;
  }
} // Rescore(point, node, oldScore)

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  // First try to find the distance bound to check if we can prune by distance.

  // Calculate the best node-to-node distance.
  const double distance = SortPolicy::BestNodeToNodeDistance(&queryNode,
                                                             &referenceNode);

  double pointBound = DBL_MAX;
  double childBound = DBL_MAX;
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  for (size_t i = 0; i < queryNode.NumPoints(); i++)
  {
    const double bound = distances(distances.n_rows - 1, queryNode.Point(i))
        + maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  for (size_t i = 0; i < queryNode.NumChildren(); i++)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (bound < childBound)
      childBound = bound;
  }

  // Update the bound.
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
  const double bestDistance = queryNode.Stat().Bound();

  return Score(queryNode, referenceNode, distance, bestDistance);
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::Score(
      TreeType& queryNode,
      TreeType& referenceNode,
      const double baseCaseResult)
{
  // First try to find the distance bound to check if we can prune
  // by distance.

  // Find the best node-to-node distance.
  const double distance = SortPolicy::BestNodeToNodeDistance(&queryNode,
      &referenceNode, baseCaseResult);

  double pointBound = DBL_MAX;
  double childBound = DBL_MAX;
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  for (size_t i = 0; i < queryNode.NumPoints(); i++)
  {
    const double bound = distances(distances.n_rows - 1, queryNode.Point(i))
        + maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  for (size_t i = 0; i < queryNode.NumChildren(); i++)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (bound < childBound)
      childBound = bound;
  }

  // update the bound
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
  const double bestDistance = queryNode.Stat().Bound();

  return Score(queryNode, referenceNode, distance, bestDistance);
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double distance,
    const double bestDistance)
{
  // Update the number of samples made for this node -- propagate up from child
  // nodes if child nodes have made samples that the parent node is not aware
  // of.  Remember, we must propagate down samples made to the child nodes if
  // 'queryNode' descend is deemed necessary.

  // Only update from children if a non-leaf node, obviously.
  if (!queryNode.IsLeaf())
  {
    size_t numSamplesMadeInChildNodes = std::numeric_limits<size_t>::max();

    // Find the minimum number of samples made among all children.
    for (size_t i = 0; i < queryNode.NumChildren(); i++)
    {
      const size_t numSamples = queryNode.Child(i).Stat().NumSamplesMade();
      if (numSamples < numSamplesMadeInChildNodes)
        numSamplesMadeInChildNodes = numSamples;
    }

    // The number of samples made for a node is propagated up from the child
    // nodes if the child nodes have made samples that the parent (which is the
    // current 'queryNode') is not aware of.
    queryNode.Stat().NumSamplesMade() = std::max(
        queryNode.Stat().NumSamplesMade(), numSamplesMadeInChildNodes);
  }

  // Now check if the node-pair interaction can be pruned.

  // If this is better than the best distance we've seen so far, maybe there
  // will be something down this node.  Also check if enough samples are already
  // made for this 'queryNode'.
  if (SortPolicy::IsBetter(distance, bestDistance)
      && queryNode.Stat().NumSamplesMade() < numSamplesReqd)
  {
    // We cannot prune this node; try approximating this node by sampling.

    // If we are required to visit the first leaf (to find possible duplicates),
    // make sure we do not approximate.
    if (queryNode.Stat().NumSamplesMade() > 0 || !firstLeafExact)
    {
      // Check if this node can be approximated by sampling.
      size_t samplesReqd = (size_t) std::ceil(samplingRatio
          * (double) referenceNode.NumDescendants());
      samplesReqd = std::min(samplesReqd, numSamplesReqd -
          queryNode.Stat().NumSamplesMade());

      if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
      {
        // If too many samples are required and we are not at a leaf, then we
        // can't prune.  Since query tree descent is necessary now, propagate
        // the number of samples made down to the children.

        // Iterate through all children and propagate the number of samples made
        // to the children.  Only update if the parent node has made samples the
        // children have not seen.
        for (size_t i = 0; i < queryNode.NumChildren(); i++)
          queryNode.Child(i).Stat().NumSamplesMade() = std::max(
              queryNode.Stat().NumSamplesMade(),
              queryNode.Child(i).Stat().NumSamplesMade());

        return distance;
      }
      else
      {
        if (!referenceNode.IsLeaf())
        {
          // Then samplesReqd <= singleSampleLimit.  Hence, approximate node by
          // sampling enough number of points for every query in the query node.
          for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
          {
            const size_t queryIndex = queryNode.Descendant(i);
            arma::uvec distinctSamples;
            RAUtil::ObtainDistinctSamples(samplesReqd,
                referenceNode.NumDescendants(), distinctSamples);
            for (size_t j = 0; j < distinctSamples.n_elem; j++)
              // The counting of the samples are done in the 'BaseCase' function
              // so no book-keeping is required here.
              BaseCase(queryIndex,
                  referenceNode.Descendant(distinctSamples[j]));
          }

          // Update the number of samples made for the queryNode and also update
          // the number of sample made for the child nodes.
          queryNode.Stat().NumSamplesMade() += samplesReqd;

          // Since we are not going to descend down the query tree for this
          // reference node, there is no point updating the number of samples
          // made for the child nodes of this query node.

          // Node is approximated, so we can prune it.
          return DBL_MAX;
        }
        else
        {
          if (sampleAtLeaves)
          {
            // Approximate node by sampling enough number of points for every
            // query in the query node.
            for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
            {
              const size_t queryIndex = queryNode.Descendant(i);
              arma::uvec distinctSamples;
              RAUtil::ObtainDistinctSamples(samplesReqd,
                  referenceNode.NumDescendants(), distinctSamples);
              for (size_t j = 0; j < distinctSamples.n_elem; j++)
                // The counting of the samples are done in the 'BaseCase'
                // function so no book-keeping is required here.
                BaseCase(queryIndex,
                    referenceNode.Descendant(distinctSamples[j]));
            }

            // Update the number of samples made for the queryNode and also
            // update the number of sample made for the child nodes.
            queryNode.Stat().NumSamplesMade() += samplesReqd;

            // Since we are not going to descend down the query tree for this
            // reference node, there is no point updating the number of samples
            // made for the child nodes of this query node.

            // (Leaf) node is approximated, so we can prune it.
            return DBL_MAX;
          }
          else
          {
            // We cannot sample from leaves, so we cannot prune.  Propagate the
            // number of samples made down to the children.

            // Go through all children and propagate the number of
            // samples made to the children.
            for (size_t i = 0; i < queryNode.NumChildren(); i++)
              queryNode.Child(i).Stat().NumSamplesMade() = std::max(
                  queryNode.Stat().NumSamplesMade(),
                  queryNode.Child(i).Stat().NumSamplesMade());

            return distance;
          }
        }
      }
    }
    else
    {
      // We must first visit the first leaf to boost accuracy.
      // Go through all children and propagate the number of
      // samples made to the children.
      for (size_t i = 0; i < queryNode.NumChildren(); i++)
        queryNode.Child(i).Stat().NumSamplesMade() = std::max(
            queryNode.Stat().NumSamplesMade(),
            queryNode.Child(i).Stat().NumSamplesMade());

      return distance;
    }
  }
  else
  {
    // Either there cannot be anything better in this node, or enough number of
    // samples are already made, so prune it.

    // Add 'fake' samples from this node; fake because the distances to
    // these samples need not be computed.  If enough samples are already made,
    // this step does not change the result of the search since this queryNode
    // will never be descended anymore.
    queryNode.Stat().NumSamplesMade() += (size_t) std::floor(samplingRatio *
        (double) referenceNode.NumDescendants());

    // Since we are not going to descend down the query tree for this reference
    // node, there is no point updating the number of samples made for the child
    // nodes of this query node.

    return DBL_MAX;
  }
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::
Rescore(TreeType& queryNode,
        TreeType& referenceNode,
        const double oldScore)
{
  if (oldScore == DBL_MAX)
    return oldScore;

  // First try to find the distance bound to check if we can prune by distance.
  double pointBound = DBL_MAX;
  double childBound = DBL_MAX;
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  for (size_t i = 0; i < queryNode.NumPoints(); i++)
  {
    const double bound = distances(distances.n_rows - 1, queryNode.Point(i))
        + maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  for (size_t i = 0; i < queryNode.NumChildren(); i++)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (bound < childBound)
      childBound = bound;
  }

  // Update the bound.
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
  const double bestDistance = queryNode.Stat().Bound();

  // Now check if the node-pair interaction can be pruned by sampling.
  // Update the number of samples made for that node.  Propagate up from child
  // nodes if child nodes have made samples that the parent node is not aware
  // of.  Remember, we must propagate down samples made to the child nodes if
  // the parent samples.

  // Only update from children if a non-leaf node, obviously.
  if (!queryNode.IsLeaf())
  {
    size_t numSamplesMadeInChildNodes = std::numeric_limits<size_t>::max();

    // Find the minimum number of samples made among all children
    for (size_t i = 0; i < queryNode.NumChildren(); i++)
    {
      const size_t numSamples = queryNode.Child(i).Stat().NumSamplesMade();
      if (numSamples < numSamplesMadeInChildNodes)
        numSamplesMadeInChildNodes = numSamples;
    }

    // The number of samples made for a node is propagated up from the child
    // nodes if the child nodes have made samples that the parent (which is the
    // current 'queryNode') is not aware of.
    queryNode.Stat().NumSamplesMade() = std::max(
        queryNode.Stat().NumSamplesMade(), numSamplesMadeInChildNodes);
  }

  // Now check if the node-pair interaction can be pruned by sampling.

  // If this is better than the best distance we've seen so far, maybe there
  // will be something down this node.  Also check if enough samples are already
  // made for this query.
  if (SortPolicy::IsBetter(oldScore, bestDistance) &&
      queryNode.Stat().NumSamplesMade() < numSamplesReqd)
  {
    // We cannot prune this node, so approximate by sampling.

    // Here we assume that since we are re-scoring, the algorithm has already
    // sampled some candidates, and if specified, also traversed to the first
    // leaf.  So no checks regarding that are made any more.
    size_t samplesReqd = (size_t) std::ceil(
        samplingRatio * (double) referenceNode.NumDescendants());
    samplesReqd  = std::min(samplesReqd,
        numSamplesReqd - queryNode.Stat().NumSamplesMade());

    if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
    {
      // If too many samples are required and we are not at a leaf, then we
      // can't prune.

      // Since query tree descent is necessary now, propagate the number of
      // samples made down to the children.

      // Go through all children and propagate the number of samples made to the
      // children.  Only update if the parent node has made samples the children
      // have not seen.
      for (size_t i = 0; i < queryNode.NumChildren(); i++)
        queryNode.Child(i).Stat().NumSamplesMade() = std::max(
            queryNode.Stat().NumSamplesMade(),
            queryNode.Child(i).Stat().NumSamplesMade());

      return oldScore;
    }
    else
    {
      if (!referenceNode.IsLeaf()) // If not a leaf,
      {
        // then samplesReqd <= singleSampleLimit.  Hence, approximate the node
        // by sampling enough points for every query in the query node.
        for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
        {
          const size_t queryIndex = queryNode.Descendant(i);
          arma::uvec distinctSamples;
          RAUtil::ObtainDistinctSamples(samplesReqd,
              referenceNode.NumDescendants(), distinctSamples);
          for (size_t j = 0; j < distinctSamples.n_elem; j++)
            // The counting of the samples are done in the 'BaseCase'
            // function so no book-keeping is required here.
            BaseCase(queryIndex, referenceNode.Descendant(distinctSamples[j]));
        }

        // Update the number of samples made for the query node and also update
        // the number of samples made for the child nodes.
        queryNode.Stat().NumSamplesMade() += samplesReqd;

        // Since we are not going to descend down the query tree for this
        // reference node, there is no point updating the number of samples made
        // for the child nodes of this query node.

        // Node approximated, so we can prune it.
        return DBL_MAX;
      }
      else // We are at a leaf.
      {
        if (sampleAtLeaves)
        {
          // Approximate node by sampling enough points for every query in the
          // query node.
          for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
          {
            const size_t queryIndex = queryNode.Descendant(i);
            arma::uvec distinctSamples;
            RAUtil::ObtainDistinctSamples(samplesReqd,
                referenceNode.NumDescendants(), distinctSamples);
            for (size_t j = 0; j < distinctSamples.n_elem; j++)
              // The counting of the samples are done in BaseCase() so no
              // book-keeping is required here.
              BaseCase(queryIndex,
                  referenceNode.Descendant(distinctSamples[j]));
          }

          // Update the number of samples made for the query node and also
          // update the number of samples made for the child nodes.
          queryNode.Stat().NumSamplesMade() += samplesReqd;

          // Since we are not going to descend down the query tree for this
          // reference node, there is no point updating the number of samples
          // made for the child nodes of this query node.

          // (Leaf) node approximated, so we can prune it.
          return DBL_MAX;
        }
        else
        {
          // We cannot sample from leaves, so we cannot prune.
          // Propagate the number of samples made down to the children.
          for (size_t i = 0; i < queryNode.NumChildren(); i++)
            queryNode.Child(i).Stat().NumSamplesMade() = std::max(
                queryNode.Stat().NumSamplesMade(),
                queryNode.Child(i).Stat().NumSamplesMade());

          return oldScore;
        }
      }
    }
  }
  else
  {
    // Either there cannot be anything better in this node, or enough samples
    // are already made, so prune it.

    // Add 'fake' samples from this node; fake because the distances to
    // these samples need not be computed.  If enough samples are already made,
    // this step does not change the result of the search since this query node
    // will never be descended anymore.
    queryNode.Stat().NumSamplesMade() += (size_t) std::floor(samplingRatio *
        (double) referenceNode.NumDescendants());

    // Since we are not going to descend down the query tree for this reference
    // node, there is no point updating the number of samples made for the child
    // nodes of this query node.
    return DBL_MAX;
  }
} // Rescore(node, node, oldScore)

/**
 * Helper function to insert a point into the neighbors and distances matrices.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param pos Position in list to insert into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
void RASearchRules<SortPolicy, MetricType, TreeType>::InsertNeighbor(
    const size_t queryIndex,
    const size_t pos,
    const size_t neighbor,
    const double distance)
{
  // We only memmove() if there is actually a need to shift something.
  if (pos < (distances.n_rows - 1))
  {
    int len = (distances.n_rows - 1) - pos;
    memmove(distances.colptr(queryIndex) + (pos + 1),
        distances.colptr(queryIndex) + pos,
        sizeof(double) * len);
    memmove(neighbors.colptr(queryIndex) + (pos + 1),
        neighbors.colptr(queryIndex) + pos,
        sizeof(size_t) * len);
  }

  // Now put the new information in the right index.
  distances(pos, queryIndex) = distance;
  neighbors(pos, queryIndex) = neighbor;
}

} // namespace neighbor
} // namespace mlpack

#endif // MLPACK_METHODS_RANN_RA_SEARCH_RULES_IMPL_HPP
