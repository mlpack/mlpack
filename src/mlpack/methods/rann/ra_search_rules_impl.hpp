/**
 * @file ra_search_rules_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of RASearchRules.
 *
 * This file is part of MLPACK 1.0.5.
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
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_RA_SEARCH_RULES_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_RA_SEARCH_RULES_IMPL_HPP

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
              const size_t singleSampleLimit) :
  referenceSet(referenceSet),
  querySet(querySet),
  neighbors(neighbors),
  distances(distances),
  metric(metric),
  sampleAtLeaves(sampleAtLeaves),
  firstLeafExact(firstLeafExact),
  singleSampleLimit(singleSampleLimit)
{ 
  Timer::Start("computing_number_of_samples_reqd");
  numSamplesReqd = MinimumSamplesReqd(referenceSet.n_cols, neighbors.n_rows,
                                      tau, alpha);
  Timer::Stop("computing_number_of_samples_reqd");

  // initializing some stats to be collected during the search
  numSamplesMade = arma::zeros<arma::Col<size_t> >(querySet.n_cols);
  numDistComputations = 0;
  samplingRatio = (double) numSamplesReqd / (double) referenceSet.n_cols;

  Log::Info << "Minimum Samples Required per-query: " << numSamplesReqd << 
    ", Sampling Ratio: " << samplingRatio << std::endl;

  if (naive) // no tree traversal going to happen, just do naive sampling here.
  {
    // sample enough number of points
    for (size_t i = 0; i < querySet.n_cols; ++i)
    {
      arma::uvec distinctSamples;
      ObtainDistinctSamples(numSamplesReqd, referenceSet.n_cols, 
                            distinctSamples);
      for (size_t j = 0; j < distinctSamples.n_elem; j++)
        BaseCase(i, (size_t) distinctSamples[j]);
    }
  }
}


template<typename SortPolicy, typename MetricType, typename TreeType>
inline force_inline
void RASearchRules<SortPolicy, MetricType, TreeType>::
ObtainDistinctSamples(const size_t numSamples,
                      const size_t rangeUpperBound,
                      arma::uvec& distinctSamples) const
{
  // keep track of the points that are sampled
  arma::Col<size_t> sampledPoints;
  sampledPoints.zeros(rangeUpperBound);

  for (size_t i = 0; i < numSamples; i++)
    sampledPoints[(size_t) math::RandInt(rangeUpperBound)]++;

  distinctSamples = arma::find(sampledPoints > 0);
  return;
}



template<typename SortPolicy, typename MetricType, typename TreeType>
size_t RASearchRules<SortPolicy, MetricType, TreeType>::
MinimumSamplesReqd(const size_t n,
                   const size_t k,
                   const double tau,
                   const double alpha) const
{
  size_t ub = n; // the upper bound on the binary search
  size_t lb = k; // the lower bound on the binary search
  size_t  m = lb; // the minimum number of random samples

  // The rank approximation
  size_t t = (size_t) std::ceil(tau * (double) n / 100.0);

  double prob;
  assert(alpha <= 1.0);

  // going through all values of sample sizes
  // to find the minimum samples required to satisfy the
  // desired bound
  bool done = false;

  // This performs a binary search on the integer values between 'lb = k' 
  // and 'ub = n' to find the minimum number of samples 'm' required to obtain
  // the desired success probability 'alpha'.
  do 
  {
    prob = SuccessProbability(n, k, m, t);

    if (prob > alpha) 
    {
      if (prob - alpha < 0.001 || ub < lb + 2) {
        done = true;
        break;
      }
      else
        ub = m;
    } 
    else 
    {
      if (prob < alpha) 
      {
        if (m == lb) 
        {
          m++;
          continue;
        }
        else 
          lb = m;
      } 
      else 
      {
        done = true;
        break;
      }
    }
    m = (ub + lb) / 2;

  } while (!done);

  return (m + 1);
}


template<typename SortPolicy, typename MetricType, typename TreeType>
double RASearchRules<SortPolicy, MetricType, TreeType>::
SuccessProbability(const size_t n,
                   const size_t k,
                   const size_t m,
                   const size_t t) const
{
  if (k == 1) 
  {
    if (m > n - t)
      return 1.0;

    double eps = (double) t / (double) n;

    return 1.0 - std::pow(1.0 - eps, (double) m);

  } // faster implementation for topK = 1
  else
  {
    if (m < k)
      return 0.0;

    if (m > n - t + k)
      return 1.0;

    double eps = (double) t / (double) n;
    double sum = 0.0;

    // The probability that 'k' of the 'm' samples lie within the top 't' 
    // of the neighbors is given by:
    // sum_{j = k}^m Choose(m, j) (t/n)^j (1 - t/n)^{m - j}
    // which is also equal to 
    // 1 - sum_{j = 0}^{k - 1} Choose(m, j) (t/n)^j (1 - t/n)^{m - j}
    //
    // So this is a m - k term summation or a k term summation. So if 
    // m > 2k, do the k term summation, otherwise do the m term summation.

    size_t lb;
    size_t ub;
    bool topHalf;

    if (2 * k < m)
    {
      // compute 1 - sum_{j = 0}^{k - 1} Choose(m, j) eps^j (1 - eps)^{m - j}
      // eps = t/n
      //
      // Choosing 'lb' as 1 and 'ub' as k so as to sum from 1 to (k - 1),
      // and add the term (1 - eps)^m term separately.
      lb = 1;
      ub = k;
      topHalf = true;
      sum = std::pow(1 - eps, (double) m);
    }
    else
    {
      // compute sum_{j = k}^m Choose(m, j) eps^j (1 - eps)^{m - j}
      // eps = t/n
      //
      // Choosing 'lb' as k and 'ub' as m so as to sum from k to (m - 1),
      // and add the term eps^m term separately.
      lb = k;
      ub = m;
      topHalf = false;
      sum = std::pow(eps, (double) m);
    }

    for (size_t j = lb; j < ub; j++)
    {
      // compute Choose(m, j) 
      double mCj = (double) m;
      size_t jTrans;

      // if j < m - j, compute Choose(m, j)
      // if j > m - j, compute Choose(m, m - j)
      if (topHalf)
        jTrans = j;
      else
        jTrans = m - j;

      for(size_t i = 2; i <= jTrans; i++) 
      {
        mCj *= (double) (m - (i - 1));
        mCj /= (double) i;
      }

      sum += (mCj * std::pow(eps, (double) j) 
              * std::pow(1.0 - eps, (double) (m - j)));
    }

    if (topHalf)
      sum = 1.0 - sum;

    return sum;
  } // for k > 1
} // FastComputeProb


template<typename SortPolicy, typename MetricType, typename TreeType>
inline force_inline 
double RASearchRules<SortPolicy, MetricType, TreeType>::
BaseCase(const size_t queryIndex, const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if ((&querySet == &referenceSet) && (queryIndex == referenceIndex))
    return 0.0;

  double distance = metric.Evaluate(querySet.unsafe_col(queryIndex),
                                    referenceSet.unsafe_col(referenceIndex));

  // If this distance is better than any of the current candidates, the
  // SortDistance() function will give us the position to insert it into.
  arma::vec queryDist = distances.unsafe_col(queryIndex);
  size_t insertPosition = SortPolicy::SortDistance(queryDist, distance);

  // SortDistance() returns (size_t() - 1) if we shouldn't add it.
  if (insertPosition != (size_t() - 1))
    InsertNeighbor(queryIndex, insertPosition, referenceIndex, distance);

  numSamplesMade[queryIndex]++;

  // TO REMOVE
  numDistComputations++;

  return distance;
}


template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::
Prescore(TreeType& queryNode,
         TreeType& referenceNode,
         TreeType& referenceChildNode,
         const double baseCaseResult) const
{ 
  return 0.0;
}


template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::
PrescoreQ(TreeType& queryNode,
          TreeType& queryChildNode,
          TreeType& referenceNode,
          const double baseCaseResult) const
{ 
  return 0.0;
}


template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::
Score(const size_t queryIndex, TreeType& referenceNode)
{
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double distance = SortPolicy::BestPointToNodeDistance(queryPoint,
      &referenceNode);
  const double bestDistance = distances(distances.n_rows - 1, queryIndex);


  // If this is better than the best distance we've seen so far, 
  // maybe there will be something down this node. 
  // Also check if enough samples are already made for this query.
  if (SortPolicy::IsBetter(distance, bestDistance)
      && numSamplesMade[queryIndex] < numSamplesReqd)
  {
    // We cannot prune this node
    // Try approximating this node by sampling

    // If you are required to visit the first leaf (to find possible 
    // duplicates), make sure you do not approximate.
    if (numSamplesMade[queryIndex] > 0 || !firstLeafExact)
    {
      // check if this node can be approximated by sampling
      size_t samplesReqd = 
        (size_t) std::ceil(samplingRatio * (double) referenceNode.Count());
      samplesReqd = std::min(samplesReqd, 
                             numSamplesReqd - numSamplesMade[queryIndex]);

      if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
      {
        // if too many samples required and not at a leaf, then can't prune
        return distance;
      }
      else
      {
        if (!referenceNode.IsLeaf()) // if not a leaf
        {
          // Then samplesReqd <= singleSampleLimit.
          // Hence approximate node by sampling enough number of points
          arma::uvec distinctSamples;
          ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                distinctSamples);
          for (size_t i = 0; i < distinctSamples.n_elem; i++)
            // The counting of the samples are done in the 'BaseCase' function
            // so no book-keeping is required here.
            BaseCase(queryIndex,
                     referenceNode.Begin() + (size_t) distinctSamples[i]);
          
          // Node approximated so we can prune it
          return DBL_MAX;
        }
        else // we are at a leaf.
        {
          if (sampleAtLeaves) // if allowed to sample at leaves.
          {
            // Approximate node by sampling enough number of points
            arma::uvec distinctSamples;
            ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                  distinctSamples);
            for (size_t i = 0; i < distinctSamples.n_elem; i++)
              // The counting of the samples are done in the 'BaseCase'
              // function so no book-keeping is required here.
              BaseCase(queryIndex,
                       referenceNode.Begin() + (size_t) distinctSamples[i]);
          
            // (Leaf) Node approximated so can prune it
            return DBL_MAX;
          }
          else 
          {
            // not allowed to sample from leaves, so cannot prune.
            return distance;
          } // sample at leaves
        } // if not-leaf
      } // if cannot-approximate by sampling
    }
    else
    {
      // try first to visit your first leaf to boost your accuracy
      // and find your (near) duplicates if they exist
      return distance;
    } // if first-leaf exact visit required
  }
  else
  {
    // Either there cannot be anything better in this node. 
    // Or enough number of samples are already made. 
    // So prune it.

    // add 'fake' samples from this node; fake because the distances to 
    // these samples need not be computed.

    // If enough samples already made, this step does not change the 
    // result of the search.
    numSamplesMade[queryIndex] += 
      (size_t) std::floor(samplingRatio * (double) referenceNode.Count());

    return DBL_MAX; 
  } // if can-prune
} // Score(point, node)


template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::
Score(const size_t queryIndex,
      TreeType& referenceNode, 
      const double baseCaseResult)
{

  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double distance = SortPolicy::BestPointToNodeDistance(queryPoint,
      &referenceNode, baseCaseResult);
  const double bestDistance = distances(distances.n_rows - 1, queryIndex);

  // Hereon, this 'Score' function is exactly same as the previous 
  // 'Score' function.

  // If this is better than the best distance we've seen so far, 
  // maybe there will be something down this node. 
  // Also check if enough samples are already made for this query.
  if (SortPolicy::IsBetter(distance, bestDistance)
      && numSamplesMade[queryIndex] < numSamplesReqd)
  {
    // We cannot prune this node
    // Try approximating this node by sampling

    // If you are required to visit the first leaf (to find possible 
    // duplicates), make sure you do not approximate.
    if (numSamplesMade[queryIndex] > 0 || !firstLeafExact)
    {
      // Check if this node can be approximated by sampling:
      // 'referenceNode.Count() should correspond to the number of points
      // present in this subtree.
      size_t samplesReqd = 
        (size_t) std::ceil(samplingRatio * (double) referenceNode.Count());
      samplesReqd = std::min(samplesReqd, 
                             numSamplesReqd - numSamplesMade[queryIndex]);

      if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
      {
        // if too many samples required and not at a leaf, then can't prune
        return distance;
      }
      else
      {
        if (!referenceNode.IsLeaf()) // if not a leaf
        {
          // Then samplesReqd <= singleSampleLimit.
          // Hence approximate node by sampling enough number of points
          // from this subtree. 
          arma::uvec distinctSamples;
          ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                distinctSamples);
          for (size_t i = 0; i < distinctSamples.n_elem; i++)
            // The counting of the samples are done in the 'BaseCase'
            // function so no book-keeping is required here.
            BaseCase(queryIndex,
                     referenceNode.Begin() + (size_t) distinctSamples[i]);
          
          // Node approximated so we can prune it
          return DBL_MAX;
        }
        else // we are at a leaf.
        {
          if (sampleAtLeaves) // if allowed to sample at leaves.
          {
            // Approximate node by sampling enough number of points
            arma::uvec distinctSamples;
            ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                  distinctSamples);
            for (size_t i = 0; i < distinctSamples.n_elem; i++)
              // The counting of the samples are done in the 'BaseCase'
              // function so no book-keeping is required here.
              BaseCase(queryIndex,
                       referenceNode.Begin() + (size_t) distinctSamples[i]);
          
            // (Leaf) Node approximated so can prune it
            return DBL_MAX;
          }
          else 
          {
            // not allowed to sample from leaves, so cannot prune.
            return distance;
          } // sample at leaves
        } // if not-leaf
      } // if cannot-approximate by sampling
    }
    else
    {
      // try first to visit your first leaf to boost your accuracy
      return distance;
    } // if first-leaf exact visit required
  }
  else
  {
    // Either there cannot be anything better in this node. 
    // Or enough number of samples are already made. 
    // So prune it.

    // add 'fake' samples from this node; fake because the distances to 
    // these samples need not be computed.

    // If enough samples already made, this step does not change the 
    // result of the search.
    if (numSamplesMade[queryIndex] < numSamplesReqd)
      // add 'fake' samples from this node; fake because the distances to 
      // these samples need not be computed.
      numSamplesMade[queryIndex] += 
        (size_t) std::floor(samplingRatio * (double) referenceNode.Count());

    return DBL_MAX; 
  } // if can-prune

  return (SortPolicy::IsBetter(distance, bestDistance)) ? distance : DBL_MAX;

} // Score(point, node, point-node-point-distance)


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
    // We cannot prune this node
    // Try approximating this node by sampling

    // Here we assume that since we are re-scoring, the algorithm 
    // has already sampled some candidates, and if specified, also 
    // traversed to the first leaf. 
    // So no checks regarding that is made any more. 
    //
    // check if this node can be approximated by sampling
    size_t samplesReqd = 
      (size_t) std::ceil(samplingRatio * (double) referenceNode.Count());
    samplesReqd = std::min(samplesReqd, 
                           numSamplesReqd - numSamplesMade[queryIndex]);

    if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
    {
      // if too many samples required and not at a leaf, then can't prune
      return oldScore;
    }
    else
    {
      if (!referenceNode.IsLeaf()) // if not a leaf
      {
        // Then samplesReqd <= singleSampleLimit.
        // Hence approximate node by sampling enough number of points
        arma::uvec distinctSamples;
        ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                              distinctSamples);
        for (size_t i = 0; i < distinctSamples.n_elem; i++)
          // The counting of the samples are done in the 'BaseCase'
          // function so no book-keeping is required here.
          BaseCase(queryIndex,
                   referenceNode.Begin() + (size_t) distinctSamples[i]);
          
        // Node approximated so we can prune it
        return DBL_MAX;
      }
      else // we are at a leaf.
      {
        if (sampleAtLeaves) // if allowed to sample at leaves.
        {
          // Approximate node by sampling enough number of points
          arma::uvec distinctSamples;
          ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                distinctSamples);
          for (size_t i = 0; i < distinctSamples.n_elem; i++)
            // The counting of the samples are done in the 'BaseCase'
            // function so no book-keeping is required here.
            BaseCase(queryIndex,
                     referenceNode.Begin() + (size_t) distinctSamples[i]);
          
          // (Leaf) Node approximated so can prune it
          return DBL_MAX;
        }
        else 
        {
          // not allowed to sample from leaves, so cannot prune.
          return oldScore;
        } // sample at leaves
      } // if not-leaf
    } // if cannot-approximate by sampling
  }
  else
  {
    // Either there cannot be anything better in this node. 
    // Or enough number of samples are already made. 
    // So prune it.

    // add 'fake' samples from this node; fake because the distances to 
    // these samples need not be computed.

    // If enough samples already made, this step does not change the 
    // result of the search.
    numSamplesMade[queryIndex] += 
      (size_t) std::floor(samplingRatio * (double) referenceNode.Count());

    return DBL_MAX; 
  } // if can-prune
} // Rescore(point, node, oldScore)


template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::
Score(TreeType& queryNode, TreeType& referenceNode)
{
  // First try to find the distance bound to check if we can prune 
  // by distance.

  // finding the best node-to-node distance
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

  // update the bound
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
  const double bestDistance = queryNode.Stat().Bound();

  // update the number of samples made for that node
  //  -- propagate up from child nodes if child nodes have made samples
  //     that the parent node is not aware of.
  //     REMEMBER to propagate down samples made to the child nodes 
  //     if 'queryNode' descend is deemed necessary.

  // only update from children if a non-leaf node obviously
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

    // The number of samples made for a node is propagated up from the 
    // child nodes if the child nodes have made samples that the parent
    // (which is the current 'queryNode') is not aware of.
    queryNode.Stat().NumSamplesMade() 
      = std::max(queryNode.Stat().NumSamplesMade(), 
                 numSamplesMadeInChildNodes);
  }

  // Now check if the node-pair interaction can be pruned

  // If this is better than the best distance we've seen so far, 
  // maybe there will be something down this node. 
  // Also check if enough samples are already made for this 'queryNode'.
  if (SortPolicy::IsBetter(distance, bestDistance)
      && queryNode.Stat().NumSamplesMade() < numSamplesReqd)
  {
    // We cannot prune this node
    // Try approximating this node by sampling

    // If you are required to visit the first leaf (to find possible 
    // duplicates), make sure you do not approximate.
    if (queryNode.Stat().NumSamplesMade() > 0 || !firstLeafExact)
    {
      // check if this node can be approximated by sampling
      size_t samplesReqd = 
        (size_t) std::ceil(samplingRatio * (double) referenceNode.Count());
      samplesReqd 
        = std::min(samplesReqd,
                   numSamplesReqd - queryNode.Stat().NumSamplesMade());

      if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
      {
        // if too many samples required and not at a leaf, then can't prune

        // Since query tree descend is necessary now, 
        // propagate the number of samples made down to the children

        // Iterate through all children and propagate the number of 
        // samples made to the children.
        // Only update if the parent node has made samples the children
        // have not seen
        for (size_t i = 0; i < queryNode.NumChildren(); i++) 
          queryNode.Child(i).Stat().NumSamplesMade() 
            = std::max(queryNode.Stat().NumSamplesMade(),
                       queryNode.Child(i).Stat().NumSamplesMade());

        return distance;
      }
      else
      {
        if (!referenceNode.IsLeaf()) // if not a leaf
        {
          // Then samplesReqd <= singleSampleLimit.
          // Hence approximate node by sampling enough number of points
          // for every query in the 'queryNode'
          for (size_t queryIndex = queryNode.Begin(); 
               queryIndex < queryNode.End(); queryIndex++)
          {
            arma::uvec distinctSamples;
            ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                  distinctSamples);
            for (size_t i = 0; i < distinctSamples.n_elem; i++)
              // The counting of the samples are done in the 'BaseCase'
              // function so no book-keeping is required here.
              BaseCase(queryIndex,
                       referenceNode.Begin() + (size_t) distinctSamples[i]);
          }

          // update the number of samples made for the queryNode and
          // also update the number of sample made for the child nodes.
          queryNode.Stat().NumSamplesMade() += samplesReqd;

          // since you are not going to descend down the query tree for this 
          // referenceNode, there is no point updating the number of 
          // samples made for the child nodes of this queryNode.

          // Node approximated so we can prune it
          return DBL_MAX;
        }
        else // we are at a leaf.
        {
          if (sampleAtLeaves) // if allowed to sample at leaves.
          {
            // Approximate node by sampling enough number of points
            // for every query in the 'queryNode'.
            for (size_t queryIndex = queryNode.Begin(); 
                 queryIndex < queryNode.End(); queryIndex++)
            {
              arma::uvec distinctSamples;
              ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                    distinctSamples);
              for (size_t i = 0; i < distinctSamples.n_elem; i++)
                // The counting of the samples are done in the 'BaseCase'
                // function so no book-keeping is required here.
                BaseCase(queryIndex,
                         referenceNode.Begin() + (size_t) distinctSamples[i]);
            }

            // update the number of samples made for the queryNode and
            // also update the number of sample made for the child nodes.
            queryNode.Stat().NumSamplesMade() += samplesReqd;

            // since you are not going to descend down the query tree for this 
            // referenceNode, there is no point updating the number of 
            // samples made for the child nodes of this queryNode.

            // (Leaf) Node approximated so can prune it
            return DBL_MAX;
          }
          else 
          {
            // Not allowed to sample from leaves, so cannot prune.
            // Propagate the number of samples made down to the children
            
            // Go through all children and propagate the number of 
            // samples made to the children.
            for (size_t i = 0; i < queryNode.NumChildren(); i++) 
              queryNode.Child(i).Stat().NumSamplesMade() 
                = std::max(queryNode.Stat().NumSamplesMade(),
                           queryNode.Child(i).Stat().NumSamplesMade());

            return distance;
          } // sample at leaves
        } // if not-leaf
      } // if cannot-approximate by sampling
    }
    else
    {
      // Have to first to visit your first leaf to boost your accuracy

      // Propagate the number of samples made down to the children

      // Go through all children and propagate the number of 
      // samples made to the children.
      for (size_t i = 0; i < queryNode.NumChildren(); i++) 
        queryNode.Child(i).Stat().NumSamplesMade() 
          = std::max(queryNode.Stat().NumSamplesMade(),
                     queryNode.Child(i).Stat().NumSamplesMade());
      
      return distance;
    } // if first-leaf exact visit required
  }
  else
  {
    // Either there cannot be anything better in this node. 
    // Or enough number of samples are already made. 
    // So prune it.

    // add 'fake' samples from this node; fake because the distances to 
    // these samples need not be computed.

    // If enough samples already made, this step does not change the 
    // result of the search since this queryNode will never be 
    // descended anymore.
    queryNode.Stat().NumSamplesMade() += 
      (size_t) std::floor(samplingRatio * (double) referenceNode.Count());

    // since you are not going to descend down the query tree for this 
    // reference node, there is no point updating the number of samples 
    // made for the child nodes of this queryNode.

    return DBL_MAX; 
  } // if can-prune
} // Score(node, node)


template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::
Score(TreeType& queryNode,
      TreeType& referenceNode, 
      const double baseCaseResult)
{
  // First try to find the distance bound to check if we can prune 
  // by distance.

  // find the best node-to-node distance
  const double distance = SortPolicy::BestNodeToNodeDistance(&queryNode,
                                                             &referenceNode,
                                                             baseCaseResult);
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

  // update the number of samples made for that node
  //  -- propagate up from child nodes if child nodes have made samples
  //     that the parent node is not aware of.
  //     REMEMBER to propagate down samples made to the child nodes 
  //     if 'queryNode' descend is deemed necessary.

  // only update from children if a non-leaf node obviously
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

    // The number of samples made for a node is propagated up from the 
    // child nodes if the child nodes have made samples that the parent
    // (which is the current 'queryNode') is not aware of.
    queryNode.Stat().NumSamplesMade() 
      = std::max(queryNode.Stat().NumSamplesMade(), 
                 numSamplesMadeInChildNodes);
  }

  // Now check if the node-pair interaction can be pruned

  // If this is better than the best distance we've seen so far, 
  // maybe there will be something down this node. 
  // Also check if enough samples are already made for this 'queryNode'.
  if (SortPolicy::IsBetter(distance, bestDistance)
      && queryNode.Stat().NumSamplesMade() < numSamplesReqd)
  {
    // We cannot prune this node
    // Try approximating this node by sampling

    // If you are required to visit the first leaf (to find possible 
    // duplicates), make sure you do not approximate.
    if (queryNode.Stat().NumSamplesMade() > 0 || !firstLeafExact)
    {
      // check if this node can be approximated by sampling
      size_t samplesReqd = 
        (size_t) std::ceil(samplingRatio * (double) referenceNode.Count());
      samplesReqd 
        = std::min(samplesReqd,
                   numSamplesReqd - queryNode.Stat().NumSamplesMade());

      if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
      {
        // if too many samples required and not at a leaf, then can't prune

        // Since query tree descend is necessary now, 
        // propagate the number of samples made down to the children

        // Iterate through all children and propagate the number of 
        // samples made to the children.
        // Only update if the parent node has made samples the children
        // have not seen
        for (size_t i = 0; i < queryNode.NumChildren(); i++) 
          queryNode.Child(i).Stat().NumSamplesMade() 
            = std::max(queryNode.Stat().NumSamplesMade(),
                       queryNode.Child(i).Stat().NumSamplesMade());

        return distance;
      }
      else
      {
        if (!referenceNode.IsLeaf()) // if not a leaf
        {
          // Then samplesReqd <= singleSampleLimit.
          // Hence approximate node by sampling enough number of points
          // for every query in the 'queryNode'
          for (size_t queryIndex = queryNode.Begin(); 
               queryIndex < queryNode.End(); queryIndex++)
          {
            arma::uvec distinctSamples;
            ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                  distinctSamples);
            for (size_t i = 0; i < distinctSamples.n_elem; i++)
              // The counting of the samples are done in the 'BaseCase'
              // function so no book-keeping is required here.
              BaseCase(queryIndex,
                       referenceNode.Begin() + (size_t) distinctSamples[i]);
          }

          // update the number of samples made for the queryNode and
          // also update the number of sample made for the child nodes.
          queryNode.Stat().NumSamplesMade() += samplesReqd;

          // since you are not going to descend down the query tree for this 
          // referenceNode, there is no point updating the number of 
          // samples made for the child nodes of this queryNode.

          // Node approximated so we can prune it
          return DBL_MAX;
        }
        else // we are at a leaf.
        {
          if (sampleAtLeaves) // if allowed to sample at leaves.
          {
            // Approximate node by sampling enough number of points
            // for every query in the 'queryNode'.
            for (size_t queryIndex = queryNode.Begin(); 
                 queryIndex < queryNode.End(); queryIndex++)
            {
              arma::uvec distinctSamples;
              ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                    distinctSamples);
              for (size_t i = 0; i < distinctSamples.n_elem; i++)
                // The counting of the samples are done in the 'BaseCase'
                // function so no book-keeping is required here.
                BaseCase(queryIndex,
                       referenceNode.Begin() + (size_t) distinctSamples[i]);
            }

            // update the number of samples made for the queryNode and
            // also update the number of sample made for the child nodes.
            queryNode.Stat().NumSamplesMade() += samplesReqd;

            // since you are not going to descend down the query tree for this 
            // referenceNode, there is no point updating the number of 
            // samples made for the child nodes of this queryNode.

            // (Leaf) Node approximated so can prune it
            return DBL_MAX;
          }
          else 
          {
            // Not allowed to sample from leaves, so cannot prune.
            // Propagate the number of samples made down to the children
            
            // Go through all children and propagate the number of 
            // samples made to the children.
            for (size_t i = 0; i < queryNode.NumChildren(); i++) 
              queryNode.Child(i).Stat().NumSamplesMade() 
                = std::max(queryNode.Stat().NumSamplesMade(),
                           queryNode.Child(i).Stat().NumSamplesMade());

            return distance;
          } // sample at leaves
        } // if not-leaf
      } // if cannot-approximate by sampling
    }
    else
    {
      // Have to first to visit your first leaf to boost your accuracy

      // Propagate the number of samples made down to the children

      // Go through all children and propagate the number of 
      // samples made to the children.
      for (size_t i = 0; i < queryNode.NumChildren(); i++) 
        queryNode.Child(i).Stat().NumSamplesMade() 
          = std::max(queryNode.Stat().NumSamplesMade(),
                     queryNode.Child(i).Stat().NumSamplesMade());
      
      return distance;
    } // if first-leaf exact visit required
  }
  else
  {
    // Either there cannot be anything better in this node. 
    // Or enough number of samples are already made. 
    // So prune it.

    // add 'fake' samples from this node; fake because the distances to 
    // these samples need not be computed.

    // If enough samples already made, this step does not change the 
    // result of the search since this queryNode will never be 
    // descended anymore.
    queryNode.Stat().NumSamplesMade() += 
      (size_t) std::floor(samplingRatio * (double) referenceNode.Count());

    // since you are not going to descend down the query tree for this 
    // reference node, there is no point updating the number of samples 
    // made for the child nodes of this queryNode.

    return DBL_MAX; 
  } // if can-prune
} // Score(node, node, baseCaseResult)


template<typename SortPolicy, typename MetricType, typename TreeType>
inline double RASearchRules<SortPolicy, MetricType, TreeType>::
Rescore(TreeType& queryNode,
        TreeType& referenceNode, 
        const double oldScore) 
{
  if (oldScore == DBL_MAX)
    return oldScore;

  // First try to find the distance bound to check if we can prune 
  // by distance.
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

  // Now check if the node-pair interaction can be pruned by sampling
  // update the number of samples made for that node
  //  -- propagate up from child nodes if child nodes have made samples
  //     that the parent node is not aware of.
  //     REMEMBER to propagate down samples made to the child nodes 
  //     if the parent samples. 

  // only update from children if a non-leaf node obviously
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

    // The number of samples made for a node is propagated up from the 
    // child nodes if the child nodes have made samples that the parent
    // (which is the current 'queryNode') is not aware of.
    queryNode.Stat().NumSamplesMade() 
      = std::max(queryNode.Stat().NumSamplesMade(), 
                 numSamplesMadeInChildNodes);
  }

  // Now check if the node-pair interaction can be pruned by sampling

  // If this is better than the best distance we've seen so far, 
  // maybe there will be something down this node. 
  // Also check if enough samples are already made for this query.
  if (SortPolicy::IsBetter(oldScore, bestDistance)
      && queryNode.Stat().NumSamplesMade() < numSamplesReqd)
  {
    // We cannot prune this node
    // Try approximating this node by sampling

    // Here we assume that since we are re-scoring, the algorithm 
    // has already sampled some candidates, and if specified, also 
    // traversed to the first leaf. 
    // So no checks regarding that is made any more. 
    //
    // check if this node can be approximated by sampling
    size_t samplesReqd = 
      (size_t) std::ceil(samplingRatio * (double) referenceNode.Count());
    samplesReqd 
      = std::min(samplesReqd,
                 numSamplesReqd - queryNode.Stat().NumSamplesMade());

    if (samplesReqd > singleSampleLimit && !referenceNode.IsLeaf())
    {
      // if too many samples required and not at a leaf, then can't prune

      // Since query tree descend is necessary now, 
      // propagate the number of samples made down to the children

      // Go through all children and propagate the number of 
      // samples made to the children.
      // Only update if the parent node has made samples the children
      // have not seen
      for (size_t i = 0; i < queryNode.NumChildren(); i++) 
        queryNode.Child(i).Stat().NumSamplesMade() 
          = std::max(queryNode.Stat().NumSamplesMade(),
                     queryNode.Child(i).Stat().NumSamplesMade());

      return oldScore;
    }
    else
    {
      if (!referenceNode.IsLeaf()) // if not a leaf
      {
        // Then samplesReqd <= singleSampleLimit.
        // Hence approximate node by sampling enough number of points
        // for every query in the 'queryNode'
        for (size_t queryIndex = queryNode.Begin(); 
             queryIndex < queryNode.End(); queryIndex++)
        {
          arma::uvec distinctSamples;
          ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                distinctSamples);
          for (size_t i = 0; i < distinctSamples.n_elem; i++)
            // The counting of the samples are done in the 'BaseCase'
            // function so no book-keeping is required here.
            BaseCase(queryIndex,
                     referenceNode.Begin() + (size_t) distinctSamples[i]);
        }
        
        // update the number of samples made for the queryNode and
        // also update the number of sample made for the child nodes.
        queryNode.Stat().NumSamplesMade() += samplesReqd;

        // since you are not going to descend down the query tree for this 
        // referenceNode, there is no point updating the number of 
        // samples made for the child nodes of this queryNode.

        // Node approximated so we can prune it
        return DBL_MAX;
      }
      else // we are at a leaf.
      {
        if (sampleAtLeaves) // if allowed to sample at leaves.
        {
          // Approximate node by sampling enough number of points
          // for every query in the 'queryNode'.
          for (size_t queryIndex = queryNode.Begin(); 
               queryIndex < queryNode.End(); queryIndex++)
          {
            arma::uvec distinctSamples;
            ObtainDistinctSamples(samplesReqd, referenceNode.Count(),
                                  distinctSamples);
            for (size_t i = 0; i < distinctSamples.n_elem; i++)
              // The counting of the samples are done in the 'BaseCase'
              // function so no book-keeping is required here.
              BaseCase(queryIndex,
                       referenceNode.Begin() + (size_t) distinctSamples[i]);
          }

          // update the number of samples made for the queryNode and
          // also update the number of sample made for the child nodes.
          queryNode.Stat().NumSamplesMade() += samplesReqd;

          // since you are not going to descend down the query tree for this 
          // referenceNode, there is no point updating the number of 
          // samples made for the child nodes of this queryNode.
          
          // (Leaf) Node approximated so can prune it
          return DBL_MAX;
        }
        else 
        {
          // not allowed to sample from leaves, so cannot prune.
          // propagate the number of samples made down to the children
            
          // going through all children and propagate the number of 
          // samples made to the children.
          for (size_t i = 0; i < queryNode.NumChildren(); i++) 
            queryNode.Child(i).Stat().NumSamplesMade() 
              = std::max(queryNode.Stat().NumSamplesMade(),
                         queryNode.Child(i).Stat().NumSamplesMade());

          return oldScore;
        } // sample at leaves
      } // if not-leaf
    } // if cannot-approximate by sampling
  }
  else
  {
    // Either there cannot be anything better in this node. 
    // Or enough number of samples are already made. 
    // So prune it.

    // add 'fake' samples from this node; fake because the distances to 
    // these samples need not be computed.

    // If enough samples already made, this step does not change the 
    // result of the search since this queryNode will never be 
    // descended anymore.
    queryNode.Stat().NumSamplesMade() += 
      (size_t) std::floor(samplingRatio * (double) referenceNode.Count());

    // since you are not going to descend down the query tree for this 
    // reference node, there is no point updating the number of samples 
    // made for the child nodes of this queryNode.

    return DBL_MAX; 
  } // if can-prune
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

}; // namespace neighbor
}; // namespace mlpack

#endif // __MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_RULES_IMPL_HPP
