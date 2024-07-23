/**
 * @file methods/rann/ra_search_rules_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of RASearchRules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANN_RA_SEARCH_RULES_IMPL_HPP
#define MLPACK_METHODS_RANN_RA_SEARCH_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "ra_search_rules.hpp"

namespace mlpack {

template<typename SortPolicy, typename DistanceType, typename TreeType>
RASearchRules<SortPolicy, DistanceType, TreeType>::
RASearchRules(const arma::mat& referenceSet,
              const arma::mat& querySet,
              const size_t k,
              DistanceType& distance,
              const double tau,
              const double alpha,
              const bool naive,
              const bool sampleAtLeaves,
              const bool firstLeafExact,
              const size_t singleSampleLimit,
              const bool sameSet) :
    referenceSet(referenceSet),
    querySet(querySet),
    k(k),
    distance(distance),
    sampleAtLeaves(sampleAtLeaves),
    firstLeafExact(firstLeafExact),
    singleSampleLimit(singleSampleLimit),
    sameSet(sameSet)
{
  // Validate tau to make sure that the rank approximation is greater than the
  // number of neighbors requested.

  // The rank approximation.
  const size_t n = referenceSet.n_cols;
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

  numSamplesReqd = RAUtil::MinimumSamplesReqd(n, k, tau, alpha);

  // Initialize some statistics to be collected during the search.
  numSamplesMade = zeros<arma::Col<size_t>>(querySet.n_cols);
  numDistComputations = 0;
  samplingRatio = (double) numSamplesReqd / (double) n;

  Log::Info << "Minimum samples required per query: " << numSamplesReqd <<
    ", sampling ratio: " << samplingRatio << std::endl;

  // Let's build the list of candidate neighbors for each query point.
  // It will be initialized with k candidates: (WorstDistance, size_t() - 1)
  // The list of candidates will be updated when visiting new points with the
  // BaseCase() method.
  const Candidate def = std::make_pair(SortPolicy::WorstDistance(),
      size_t() - 1);

  std::vector<Candidate> vect(k, def);
  CandidateList pqueue(CandidateCmp(), std::move(vect));

  candidates.reserve(querySet.n_cols);
  for (size_t i = 0; i < querySet.n_cols; ++i)
    candidates.push_back(pqueue);

  if (naive) // No tree traversal; just do naive sampling here.
  {
    // Sample enough points.
    arma::uvec distinctSamples;
    for (size_t i = 0; i < querySet.n_cols; ++i)
    {
      distinctSamples = arma::randperm(n, numSamplesReqd);
      for (size_t j = 0; j < distinctSamples.n_elem; ++j)
        BaseCase(i, (size_t) distinctSamples[j]);
    }
  }
}

template<typename SortPolicy, typename DistanceType, typename TreeType>
void RASearchRules<SortPolicy, DistanceType, TreeType>::GetResults(
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
{
  neighbors.set_size(k, querySet.n_cols);
  distances.set_size(k, querySet.n_cols);

  for (size_t i = 0; i < querySet.n_cols; ++i)
  {
    CandidateList& pqueue = candidates[i];
    for (size_t j = 1; j <= k; ++j)
    {
      neighbors(k - j, i) = pqueue.top().second;
      distances(k - j, i) = pqueue.top().first;
      pqueue.pop();
    }
  }
};

template<typename SortPolicy, typename DistanceType, typename TreeType>
inline mlpack_force_inline
double RASearchRules<SortPolicy, DistanceType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if (sameSet && (queryIndex == referenceIndex))
    return 0.0;

  double d = distance.Evaluate(querySet.unsafe_col(queryIndex),
                               referenceSet.unsafe_col(referenceIndex));

  InsertNeighbor(queryIndex, referenceIndex, d);

  numSamplesMade[queryIndex]++;

  numDistComputations++;

  return d;
}

template<typename SortPolicy, typename DistanceType, typename TreeType>
inline double RASearchRules<SortPolicy, DistanceType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double d = SortPolicy::BestPointToNodeDistance(queryPoint,
      &referenceNode);
  const double bestDistance = candidates[queryIndex].top().first;

  return Score(queryIndex, referenceNode, d, bestDistance);
}

template<typename SortPolicy, typename DistanceType, typename TreeType>
inline double RASearchRules<SortPolicy, DistanceType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double baseCaseResult)
{
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double d = SortPolicy::BestPointToNodeDistance(queryPoint,
      &referenceNode, baseCaseResult);
  const double bestDistance = candidates[queryIndex].top().first;

  return Score(queryIndex, referenceNode, d, bestDistance);
}

template<typename SortPolicy, typename DistanceType, typename TreeType>
inline double RASearchRules<SortPolicy, DistanceType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double dist,
    const double bestDistance)
{
  // If this is better than the best distance we've seen so far, maybe there
  // will be something down this node.  Also check if enough samples are already
  // made for this query.
  if (SortPolicy::IsBetter(dist, bestDistance)
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
        return dist;
      }
      else
      {
        if (!referenceNode.IsLeaf())
        {
          // Then samplesReqd <= singleSampleLimit.
          // Hence, approximate the node by sampling enough number of points.
          arma::uvec distinctSamples =
              arma::randperm(referenceNode.NumDescendants(), samplesReqd);
          for (size_t i = 0; i < distinctSamples.n_elem; ++i)
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
            arma::uvec distinctSamples =
                arma::randperm(referenceNode.NumDescendants(), samplesReqd);
            for (size_t i = 0; i < distinctSamples.n_elem; ++i)
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
            return dist;
          }
        }
      }
    }
    else
    {
      // Try first to visit the first leaf to boost your accuracy and find
      // (near) duplicates if they exist.
      return dist;
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

template<typename SortPolicy, typename DistanceType, typename TreeType>
inline double RASearchRules<SortPolicy, DistanceType, TreeType>::
Rescore(const size_t queryIndex,
        TreeType& referenceNode,
        const double oldScore)
{
  // If we are already pruning, still prune.
  if (oldScore == DBL_MAX)
    return oldScore;

  // Just check the score again against the distances.
  const double bestDistance = candidates[queryIndex].top().first;

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
        arma::uvec distinctSamples =
            arma::randperm(referenceNode.NumDescendants(), samplesReqd);
        for (size_t i = 0; i < distinctSamples.n_elem; ++i)
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
          arma::uvec distinctSamples =
              arma::randperm(referenceNode.NumDescendants(), samplesReqd);
          for (size_t i = 0; i < distinctSamples.n_elem; ++i)
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

template<typename SortPolicy, typename DistanceType, typename TreeType>
inline double RASearchRules<SortPolicy, DistanceType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  // First try to find the distance bound to check if we can prune by distance.

  // Calculate the best node-to-node distance.
  const double dist = SortPolicy::BestNodeToNodeDistance(&queryNode,
                                                         &referenceNode);

  double pointBound = DBL_MAX;
  double childBound = DBL_MAX;
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const double bound = candidates[queryNode.Point(i)].top().first
        + maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (bound < childBound)
      childBound = bound;
  }

  // Update the bound.
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
  const double bestDistance = queryNode.Stat().Bound();

  return Score(queryNode, referenceNode, dist, bestDistance);
}

template<typename SortPolicy, typename DistanceType, typename TreeType>
inline double RASearchRules<SortPolicy, DistanceType, TreeType>::Score(
      TreeType& queryNode,
      TreeType& referenceNode,
      const double baseCaseResult)
{
  // First try to find the distance bound to check if we can prune
  // by distance.

  // Find the best node-to-node distance.
  const double dist = SortPolicy::BestNodeToNodeDistance(&queryNode,
      &referenceNode, baseCaseResult);

  double pointBound = DBL_MAX;
  double childBound = DBL_MAX;
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const double bound = candidates[queryNode.Point(i)].top().first
        + maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (bound < childBound)
      childBound = bound;
  }

  // update the bound
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
  const double bestDistance = queryNode.Stat().Bound();

  return Score(queryNode, referenceNode, dist, bestDistance);
}

template<typename SortPolicy, typename DistanceType, typename TreeType>
inline double RASearchRules<SortPolicy, DistanceType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double dist,
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
    for (size_t i = 0; i < queryNode.NumChildren(); ++i)
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
  if (SortPolicy::IsBetter(dist, bestDistance)
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
        for (size_t i = 0; i < queryNode.NumChildren(); ++i)
          queryNode.Child(i).Stat().NumSamplesMade() = std::max(
              queryNode.Stat().NumSamplesMade(),
              queryNode.Child(i).Stat().NumSamplesMade());

        return dist;
      }
      else
      {
        if (!referenceNode.IsLeaf())
        {
          // Then samplesReqd <= singleSampleLimit.  Hence, approximate node by
          // sampling enough number of points for every query in the query node.
          arma::uvec distinctSamples;
          for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
          {
            const size_t queryIndex = queryNode.Descendant(i);
            distinctSamples = arma::randperm(referenceNode.NumDescendants(),
                samplesReqd);
            for (size_t j = 0; j < distinctSamples.n_elem; ++j)
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
            arma::uvec distinctSamples;
            for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
            {
              const size_t queryIndex = queryNode.Descendant(i);
              distinctSamples = arma::randperm(referenceNode.NumDescendants(),
                  samplesReqd);
              for (size_t j = 0; j < distinctSamples.n_elem; ++j)
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
            for (size_t i = 0; i < queryNode.NumChildren(); ++i)
              queryNode.Child(i).Stat().NumSamplesMade() = std::max(
                  queryNode.Stat().NumSamplesMade(),
                  queryNode.Child(i).Stat().NumSamplesMade());

            return dist;
          }
        }
      }
    }
    else
    {
      // We must first visit the first leaf to boost accuracy.
      // Go through all children and propagate the number of
      // samples made to the children.
      for (size_t i = 0; i < queryNode.NumChildren(); ++i)
        queryNode.Child(i).Stat().NumSamplesMade() = std::max(
            queryNode.Stat().NumSamplesMade(),
            queryNode.Child(i).Stat().NumSamplesMade());

      return dist;
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

template<typename SortPolicy, typename DistanceType, typename TreeType>
inline double RASearchRules<SortPolicy, DistanceType, TreeType>::
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

  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const double bound = candidates[queryNode.Point(i)].top().first
        + maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
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
    for (size_t i = 0; i < queryNode.NumChildren(); ++i)
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
      for (size_t i = 0; i < queryNode.NumChildren(); ++i)
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
        arma::uvec distinctSamples;
        for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
        {
          const size_t queryIndex = queryNode.Descendant(i);
          distinctSamples = arma::randperm(referenceNode.NumDescendants(),
              samplesReqd);
          for (size_t j = 0; j < distinctSamples.n_elem; ++j)
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
          arma::uvec distinctSamples;
          for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
          {
            const size_t queryIndex = queryNode.Descendant(i);
            distinctSamples = arma::randperm(referenceNode.NumDescendants(),
                samplesReqd);
            for (size_t j = 0; j < distinctSamples.n_elem; ++j)
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
          for (size_t i = 0; i < queryNode.NumChildren(); ++i)
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
 * Helper function to insert a point into the list of candidate points.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param neighbor Index of reference point which is being inserted.
 * @param dist Distance from query point to reference point.
 */
template<typename SortPolicy, typename DistanceType, typename TreeType>
inline void RASearchRules<SortPolicy, DistanceType, TreeType>::
InsertNeighbor(
    const size_t queryIndex,
    const size_t neighbor,
    const double dist)
{
  CandidateList& pqueue = candidates[queryIndex];
  Candidate c = std::make_pair(dist, neighbor);

  if (CandidateCmp()(c, pqueue.top()))
  {
    pqueue.pop();
    pqueue.push(c);
  }
}

} // namespace mlpack

#endif // MLPACK_METHODS_RANN_RA_SEARCH_RULES_IMPL_HPP
