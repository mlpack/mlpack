/**
 * @file range_search_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the RangeSearch class.
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
#ifndef __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_IMPL_HPP
#define __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_IMPL_HPP

// Just in case it hasn't been included.
#include "range_search.hpp"

namespace mlpack {
namespace range {

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
    const typename TreeType::Mat& referenceSet,
    const typename TreeType::Mat& querySet,
    const bool naive,
    const bool singleMode,
    const size_t leafSize,
    const MetricType metric) :
    referenceCopy(referenceSet),
    queryCopy(querySet),
    referenceSet(referenceCopy),
    querySet(queryCopy),
    ownReferenceTree(true),
    ownQueryTree(true),
    naive(naive),
    singleMode(!naive && singleMode), // Naive overrides single mode.
    metric(metric),
    numberOfPrunes(0)
{
  // Build the trees.
  Timer::Start("range_search/tree_building");

  // Naive sets the leaf size such that the entire tree is one node.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceCopy.n_cols : leafSize));

  queryTree = new TreeType(queryCopy, oldFromNewQueries,
      (naive ? queryCopy.n_cols : leafSize));

  Timer::Stop("range_search/tree_building");
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
    const typename TreeType::Mat& referenceSet,
    const bool naive,
    const bool singleMode,
    const size_t leafSize,
    const MetricType metric) :
    referenceCopy(referenceSet),
    referenceSet(referenceCopy),
    querySet(referenceCopy),
    queryTree(NULL),
    ownReferenceTree(true),
    ownQueryTree(false),
    naive(naive),
    singleMode(!naive && singleMode), // Naive overrides single mode.
    metric(metric),
    numberOfPrunes(0)
{
  // Build the trees.
  Timer::Start("range_search/tree_building");

  // Naive sets the leaf size such that the entire tree is one node.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceCopy.n_cols : leafSize));

  Timer::Stop("range_search/tree_building");
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
    TreeType* referenceTree,
    TreeType* queryTree,
    const typename TreeType::Mat& referenceSet,
    const typename TreeType::Mat& querySet,
    const bool singleMode,
    const MetricType metric) :
    referenceSet(referenceSet),
    querySet(querySet),
    referenceTree(referenceTree),
    queryTree(queryTree),
    ownReferenceTree(false),
    ownQueryTree(false),
    naive(false),
    singleMode(singleMode),
    metric(metric),
    numberOfPrunes(0)
{
  // Nothing else to initialize.
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
    TreeType* referenceTree,
    const typename TreeType::Mat& referenceSet,
    const bool singleMode,
    const MetricType metric) :
    referenceSet(referenceSet),
    querySet(referenceSet),
    referenceTree(referenceTree),
    queryTree(NULL),
    ownReferenceTree(false),
    ownQueryTree(false),
    naive(false),
    singleMode(singleMode),
    metric(metric),
    numberOfPrunes(0)
{
  // Nothing else to initialize.
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::~RangeSearch()
{
  if (ownReferenceTree)
    delete referenceTree;
  if (ownQueryTree)
    delete queryTree;
}

template<typename MetricType, typename TreeType>
void RangeSearch<MetricType, TreeType>::Search(
    const math::Range& range,
    std::vector<std::vector<size_t> >& neighbors,
    std::vector<std::vector<double> >& distances)
{
  Timer::Start("range_search/computing_neighbors");

  // Set size of prunes to 0.
  numberOfPrunes = 0;

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid extra copies, we will store the unmapped neighbors and distances
  // in a separate object.
  std::vector<std::vector<size_t> >* neighborPtr = &neighbors;
  std::vector<std::vector<double> >* distancePtr = &distances;

  if (ownQueryTree || (ownReferenceTree && !queryTree))
    distancePtr = new std::vector<std::vector<double> >;
  if (ownReferenceTree || ownQueryTree)
    neighborPtr = new std::vector<std::vector<size_t> >;

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(querySet.n_cols);
  distancePtr->clear();
  distancePtr->resize(querySet.n_cols);

  if (naive)
  {
    // Run the base case.
    if (!queryTree)
      ComputeBaseCase(referenceTree, referenceTree, range, *neighborPtr,
          *distancePtr);
    else
      ComputeBaseCase(referenceTree, queryTree, range, *neighborPtr,
          *distancePtr);
  }
  else if (singleMode)
  {
    // Loop over each of the query points.
    for (size_t i = 0; i < querySet.n_cols; i++)
    {
      SingleTreeRecursion(referenceTree, querySet.col(i), i, range,
          (*neighborPtr)[i], (*distancePtr)[i]);
    }
  }
  else
  {
    if (!queryTree) // References are the same as queries.
      DualTreeRecursion(referenceTree, referenceTree, range, *neighborPtr,
          *distancePtr);
    else
      DualTreeRecursion(referenceTree, queryTree, range, *neighborPtr,
          *distancePtr);
  }

  Timer::Stop("range_search/computing_neighbors");

  // Output number of prunes.
  Log::Info << "Number of pruned nodes during computation: " << numberOfPrunes
      << "." << std::endl;

  // Map points back to original indices, if necessary.
  if (!ownReferenceTree && !ownQueryTree)
  {
    // No mapping needed.  We are done.
    return;
  }
  else if (ownReferenceTree && ownQueryTree) // Map references and queries.
  {
    neighbors.clear();
    neighbors.resize(querySet.n_cols);
    distances.clear();
    distances.resize(querySet.n_cols);

    for (size_t i = 0; i < distances.size(); i++)
    {
      // Map distances (copy a column).
      size_t queryMapping = oldFromNewQueries[i];
      distances[queryMapping] = (*distancePtr)[i];

      // Copy each neighbor individually, because we need to map it.
      neighbors[queryMapping].resize(distances[queryMapping].size());
      for (size_t j = 0; j < distances[queryMapping].size(); j++)
      {
        neighbors[queryMapping][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
      }
    }

    // Finished with temporary objects.
    delete neighborPtr;
    delete distancePtr;
  }
  else if (ownReferenceTree)
  {
    if (!queryTree) // No query tree -- map both references and queries.
    {
      neighbors.clear();
      neighbors.resize(querySet.n_cols);
      distances.clear();
      distances.resize(querySet.n_cols);

      for (size_t i = 0; i < distances.size(); i++)
      {
        // Map distances (copy a column).
        size_t refMapping = oldFromNewReferences[i];
        distances[refMapping] = (*distancePtr)[i];

        // Copy each neighbor individually, because we need to map it.
        neighbors[refMapping].resize(distances[refMapping].size());
        for (size_t j = 0; j < distances[refMapping].size(); j++)
        {
          neighbors[refMapping][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
        }
      }

      // Finished with temporary objects.
      delete neighborPtr;
      delete distancePtr;
    }
    else // Map only references.
    {
      neighbors.clear();
      neighbors.resize(querySet.n_cols);

      // Map indices of neighbors.
      for (size_t i = 0; i < neighbors.size(); i++)
      {
        neighbors[i].resize((*neighborPtr)[i].size());
        for (size_t j = 0; j < neighbors[i].size(); j++)
        {
          neighbors[i][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
        }
      }

      // Finished with temporary object.
      delete neighborPtr;
    }
  }
  else if (ownQueryTree)
  {
    neighbors.clear();
    neighbors.resize(querySet.n_cols);
    distances.clear();
    distances.resize(querySet.n_cols);

    for (size_t i = 0; i < distances.size(); i++)
    {
      // Map distances (copy a column).
      distances[oldFromNewQueries[i]] = (*distancePtr)[i];

      // Map neighbors.
      neighbors[oldFromNewQueries[i]] = (*neighborPtr)[i];
    }

    // Finished with temporary objects.
    delete neighborPtr;
    delete distancePtr;
  }
}

template<typename MetricType, typename TreeType>
void RangeSearch<MetricType, TreeType>::ComputeBaseCase(
    const TreeType* referenceNode,
    const TreeType* queryNode,
    const math::Range& range,
    std::vector<std::vector<size_t> >& neighbors,
    std::vector<std::vector<double> >& distances) const
{
  // node->Begin() is the index of the first point in the node,
  // node->End() is one past the last index.
  for (size_t queryIndex = queryNode->Begin(); queryIndex < queryNode->End();
       queryIndex++)
  {
    double minDistance =
        referenceNode->Bound().MinDistance(querySet.col(queryIndex));
    double maxDistance =
        referenceNode->Bound().MaxDistance(querySet.col(queryIndex));

    // Now see if any points could fall into the range.
    if (range.Contains(math::Range(minDistance, maxDistance)))
    {
      // Loop through the reference points and see which fall into the range.
      for (size_t referenceIndex = referenceNode->Begin();
          referenceIndex < referenceNode->End(); referenceIndex++)
      {
        // We can't add points that are ourselves.
        if (referenceNode != queryNode || referenceIndex != queryIndex)
        {
          double distance = metric.Evaluate(querySet.col(queryIndex),
                                            referenceSet.col(referenceIndex));

          // If this lies in the range, add it.
          if (range.Contains(distance))
          {
            neighbors[queryIndex].push_back(referenceIndex);
            distances[queryIndex].push_back(distance);
          }
        }
      }
    }
  }
}

template<typename MetricType, typename TreeType>
void RangeSearch<MetricType, TreeType>::DualTreeRecursion(
    const TreeType* referenceNode,
    const TreeType* queryNode,
    const math::Range& range,
    std::vector<std::vector<size_t> >& neighbors,
    std::vector<std::vector<double> >& distances)
{
  // See if we can prune this node.
  math::Range distance =
      referenceNode->Bound().RangeDistance(queryNode->Bound());

  if (!range.Contains(distance))
  {
    numberOfPrunes++; // Don't recurse.  These nodes can't contain anything.
    return;
  }

  // If both nodes are leaves, then we compute the base case.
  if (referenceNode->IsLeaf() && queryNode->IsLeaf())
  {
    ComputeBaseCase(referenceNode, queryNode, range, neighbors, distances);
  }
  else if (referenceNode->IsLeaf())
  {
    // We must descend down the query node to get a leaf.
    DualTreeRecursion(referenceNode, queryNode->Left(), range, neighbors,
        distances);
    DualTreeRecursion(referenceNode, queryNode->Right(), range, neighbors,
        distances);
  }
  else if (queryNode->IsLeaf())
  {
    // We must descend down the reference node to get a leaf.
    DualTreeRecursion(referenceNode->Left(), queryNode, range, neighbors,
        distances);
    DualTreeRecursion(referenceNode->Right(), queryNode, range, neighbors,
        distances);
  }
  else
  {
    // First descend the left reference node.
    DualTreeRecursion(referenceNode->Left(), queryNode->Left(), range,
        neighbors, distances);
    DualTreeRecursion(referenceNode->Left(), queryNode->Right(), range,
        neighbors, distances);

    // Now descend the right reference node.
    DualTreeRecursion(referenceNode->Right(), queryNode->Left(), range,
        neighbors, distances);
    DualTreeRecursion(referenceNode->Right(), queryNode->Right(), range,
        neighbors, distances);
  }
}

template<typename MetricType, typename TreeType>
template<typename VecType>
void RangeSearch<MetricType, TreeType>::SingleTreeRecursion(
    const TreeType* referenceNode,
    const VecType& queryPoint,
    const size_t queryIndex,
    const math::Range& range,
    std::vector<size_t>& neighbors,
    std::vector<double>& distances)
{
  // See if we need to recurse or if we can perform base-case computations.
  if (referenceNode->IsLeaf())
  {
    // Base case: reference node is a leaf.
    for (size_t referenceIndex = referenceNode->Begin(); referenceIndex !=
         referenceNode->End(); referenceIndex++)
    {
      // Don't add this point if it is the same as the query point.
      if (!queryTree && !(referenceIndex == queryIndex))
      {
        double distance = metric.Evaluate(queryPoint,
                                          referenceSet.col(referenceIndex));

        // See if the point is in the range we are looking for.
        if (range.Contains(distance))
        {
          neighbors.push_back(referenceIndex);
          distances.push_back(distance);
        }
      }
    }
  }
  else
  {
    // Recurse down the tree.
    math::Range distanceLeft =
        referenceNode->Left()->Bound().RangeDistance(queryPoint);
    math::Range distanceRight =
        referenceNode->Right()->Bound().RangeDistance(queryPoint);

    if (range.Contains(distanceLeft))
    {
      // The left may have points we want to recurse to.
      SingleTreeRecursion(referenceNode->Left(), queryPoint, queryIndex,
          range, neighbors, distances);
    }
    else
    {
      numberOfPrunes++;
    }

    if (range.Contains(distanceRight))
    {
      // The right may have points we want to recurse to.
      SingleTreeRecursion(referenceNode->Right(), queryPoint, queryIndex,
          range, neighbors, distances);
    }
    else
    {
      numberOfPrunes++;
    }
  }
}

}; // namespace range
}; // namespace mlpack

#endif
