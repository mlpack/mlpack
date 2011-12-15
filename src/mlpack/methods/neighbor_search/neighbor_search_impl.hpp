/**
 * @file neighbor_search_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Neighbor-Search class to perform all-nearest-neighbors on
 * two specified data sets.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>

using namespace mlpack::neighbor;

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::
NeighborSearch(const typename TreeType::Mat& referenceSet,
               const typename TreeType::Mat& querySet,
               const bool naive,
               const bool singleMode,
               const size_t leafSize,
               const MetricType metric) :
    referenceCopy(referenceSet),
    queryCopy(querySet),
    referenceSet(referenceCopy),
    querySet(queryCopy),
    referenceTree(NULL),
    queryTree(NULL),
    ownReferenceTree(true), // False if a tree was passed.
    ownQueryTree(true), // False if a tree was passed.
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric),
    numberOfPrunes(0)
{
  // C++11 will allow us to call out to other constructors so we can avoid this
  // copypasta problem.

  // We'll time tree building, but only if we are building trees.
  if (!referenceTree || !queryTree)
    Timer::Start("tree_building");

  // Construct as a naive object if we need to.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceCopy.n_cols : leafSize));

  queryTree = new TreeType(queryCopy, oldFromNewQueries,
      (naive ? querySet.n_cols : leafSize));

  // Stop the timer we started above (if we need to).
  if (!referenceTree || !queryTree)
    Timer::Stop("tree_building");
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::
NeighborSearch(const typename TreeType::Mat& referenceSet,
               const bool naive,
               const bool singleMode,
               const size_t leafSize,
               const MetricType metric) :
    referenceCopy(referenceSet),
    referenceSet(referenceCopy),
    querySet(referenceCopy),
    referenceTree(NULL),
    queryTree(NULL),
    ownReferenceTree(true),
    ownQueryTree(false), // Since it will be the same as referenceTree.
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric),
    numberOfPrunes(0)
{
  // We'll time tree building, but only if we are building trees.
  Timer::Start("tree_building");

  // Construct as a naive object if we need to.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceSet.n_cols : leafSize));

  // Stop the timer we started above.
  Timer::Stop("tree_building");
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::NeighborSearch(
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

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::NeighborSearch(
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

/**
 * The tree is the only member we may be responsible for deleting.  The others
 * will take care of themselves.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::~NeighborSearch()
{
  if (ownReferenceTree)
    delete referenceTree;
  if (ownQueryTree)
    delete queryTree;
}

/**
 * Computes the best neighbors and stores them in resultingNeighbors and
 * distances.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
void NeighborSearch<SortPolicy, MetricType, TreeType>::Search(
    const size_t k,
    arma::Mat<size_t>& resultingNeighbors,
    arma::mat& distances)
{
  Timer::Start("computing_neighbors");

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid an extra copy, we will store the neighbors and distances in a
  // separate matrix.
  arma::Mat<size_t>* neighborPtr = &resultingNeighbors;
  arma::mat* distancePtr = &distances;

  if (ownQueryTree || (ownReferenceTree && !queryTree))
    distancePtr = new arma::mat; // Query indices need to be mapped.
  if (ownReferenceTree || ownQueryTree)
    neighborPtr = new arma::Mat<size_t>; // All indices need mapping.

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());

  if (naive)
  {
    // Run the base case computation on all nodes
    if (queryTree)
      ComputeBaseCase(queryTree, referenceTree, *neighborPtr, *distancePtr);
    else
      ComputeBaseCase(referenceTree, referenceTree, *neighborPtr, *distancePtr);
  }
  else
  {
    if (singleMode)
    {
      // Do one tenth of the query set at a time.
      size_t chunk = querySet.n_cols / 10;

      for (size_t i = 0; i < 10; i++)
      {
        for (size_t j = 0; j < chunk; j++)
        {
          double worstDistance = SortPolicy::WorstDistance();
          ComputeSingleNeighborsRecursion(i * chunk + j,
              querySet.unsafe_col(i * chunk + j), referenceTree, worstDistance,
              *neighborPtr, *distancePtr);
        }
      }

      // The last tenth is differently sized...
      for (size_t i = 0; i < querySet.n_cols % 10; i++)
      {
        size_t ind = (querySet.n_cols / 10) * 10 + i;
        double worstDistance = SortPolicy::WorstDistance();
        ComputeSingleNeighborsRecursion(ind, querySet.unsafe_col(ind),
            referenceTree, worstDistance, *neighborPtr, *distancePtr);
      }
    }
    else // Dual-tree recursion.
    {
      // Start on the root of each tree.
      if (queryTree)
      {
        ComputeDualNeighborsRecursion(queryTree, referenceTree,
            SortPolicy::BestNodeToNodeDistance(queryTree, referenceTree),
            *neighborPtr, *distancePtr);
      }
      else
      {
        ComputeDualNeighborsRecursion(referenceTree, referenceTree,
            SortPolicy::BestNodeToNodeDistance(referenceTree, referenceTree),
            *neighborPtr, *distancePtr);
      }
    }
  }

  Timer::Stop("computing_neighbors");

  // Now, do we need to do mapping of indices?
  if (!ownReferenceTree && !ownQueryTree)
  {
    // No mapping needed.  We are done.
    return;
  }
  else if (ownReferenceTree && ownQueryTree) // Map references and queries.
  {
    // Set size of output matrices correctly.
    resultingNeighbors.set_size(k, querySet.n_cols);
    distances.set_size(k, querySet.n_cols);

    for (size_t i = 0; i < distances.n_cols; i++)
    {
      // Map distances (copy a column).
      distances.col(oldFromNewQueries[i]) = distancePtr->col(i);

      // Map indices of neighbors.
      for (size_t j = 0; j < distances.n_rows; j++)
      {
        resultingNeighbors(j, oldFromNewQueries[i]) =
            oldFromNewReferences[(*neighborPtr)(j, i)];
      }
    }

    // Finished with temporary matrices.
    delete neighborPtr;
    delete distancePtr;
  }
  else if (ownReferenceTree)
  {
    if (!queryTree) // No query tree -- map both references and queries.
    {
      resultingNeighbors.set_size(k, querySet.n_cols);
      distances.set_size(k, querySet.n_cols);

      for (size_t i = 0; i < distances.n_cols; i++)
      {
        // Map distances (copy a column).
        distances.col(oldFromNewReferences[i]) = distancePtr->col(i);

        // Map indices of neighbors.
        for (size_t j = 0; j < distances.n_rows; j++)
        {
          resultingNeighbors(j, oldFromNewReferences[i]) =
              oldFromNewReferences[(*neighborPtr)(j, i)];
        }
      }
    }
    else // Map only references.
    {
      // Set size of neighbor indices matrix correctly.
      resultingNeighbors.set_size(k, querySet.n_cols);

      // Map indices of neighbors.
      for (size_t i = 0; i < resultingNeighbors.n_cols; i++)
      {
        for (size_t j = 0; j < resultingNeighbors.n_rows; j++)
        {
          resultingNeighbors(j, i) = oldFromNewReferences[(*neighborPtr)(j, i)];
        }
      }
    }

    // Finished with temporary matrix.
    delete neighborPtr;
  }
  else if (ownQueryTree)
  {
    // Set size of matrices correctly.
    resultingNeighbors.set_size(k, querySet.n_cols);
    distances.set_size(k, querySet.n_cols);

    for (size_t i = 0; i < distances.n_cols; i++)
    {
      // Map distances (copy a column).
      distances.col(oldFromNewQueries[i]) = distancePtr->col(i);

      // Map indices of neighbors.
      resultingNeighbors.col(oldFromNewQueries[i]) = neighborPtr->col(i);
    }

    // Finished with temporary matrices.
    delete neighborPtr;
    delete distancePtr;
  }
} // Search

/**
 * Performs exhaustive computation between two leaves.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
void NeighborSearch<SortPolicy, MetricType, TreeType>::ComputeBaseCase(
      TreeType* queryNode,
      TreeType* referenceNode,
      arma::Mat<size_t>& neighbors,
      arma::mat& distances)
{
  // Used to find the query node's new upper bound.
  double queryWorstDistance = SortPolicy::BestDistance();

  // node->Begin() is the index of the first point in the node,
  // node->End() is one past the last index.
  for (size_t queryIndex = queryNode->Begin(); queryIndex < queryNode->End();
       queryIndex++)
  {
    // Get the query point from the matrix.
    arma::vec queryPoint = querySet.unsafe_col(queryIndex);

    double queryToNodeDistance =
        SortPolicy::BestPointToNodeDistance(queryPoint, referenceNode);

    if (SortPolicy::IsBetter(queryToNodeDistance,
        distances(distances.n_rows - 1, queryIndex)))
    {
      // We'll do the same for the references.
      for (size_t referenceIndex = referenceNode->Begin();
          referenceIndex < referenceNode->End(); referenceIndex++)
      {
        // Confirm that points do not identify themselves as neighbors
        // in the monochromatic case.
        if (referenceNode != queryNode || referenceIndex != queryIndex)
        {
          arma::vec referencePoint = referenceSet.unsafe_col(referenceIndex);

          double distance = metric.Evaluate(queryPoint, referencePoint);

          // If the reference point is closer than any of the current
          // candidates, add it to the list.
          arma::vec queryDist = distances.unsafe_col(queryIndex);
          size_t insertPosition = SortPolicy::SortDistance(queryDist,
              distance);

          if (insertPosition != (size_t() - 1))
            InsertNeighbor(queryIndex, insertPosition, referenceIndex,
                distance, neighbors, distances);
        }
      }
    }

    // We need to find the upper bound distance for this query node
    if (SortPolicy::IsBetter(queryWorstDistance,
        distances(distances.n_rows - 1, queryIndex)))
      queryWorstDistance = distances(distances.n_rows - 1, queryIndex);
  }

  // Update the upper bound for the queryNode
  queryNode->Stat().Bound() = queryWorstDistance;

} // ComputeBaseCase()

/**
 * The recursive function for dual tree.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
void NeighborSearch<SortPolicy, MetricType, TreeType>::
ComputeDualNeighborsRecursion(
    TreeType* queryNode,
    TreeType* referenceNode,
    const double lowerBound,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
{
  if (SortPolicy::IsBetter(queryNode->Stat().Bound(), lowerBound))
  {
    numberOfPrunes++; // Pruned by distance; the nodes cannot be any closer
    return;           // than the already established lower bound.
  }

  if (queryNode->IsLeaf() && referenceNode->IsLeaf())
  {
    // Base case: both are leaves.
    ComputeBaseCase(queryNode, referenceNode, neighbors, distances);
    return;
  }

  if (queryNode->IsLeaf())
  {
    // We must keep descending down the reference node to get to a leaf.

    // We'll order the computation by distance; descend in the direction of less
    // distance first.
    double leftDistance = SortPolicy::BestNodeToNodeDistance(queryNode,
        referenceNode->Left());
    double rightDistance = SortPolicy::BestNodeToNodeDistance(queryNode,
        referenceNode->Right());

    if (SortPolicy::IsBetter(leftDistance, rightDistance))
    {
      ComputeDualNeighborsRecursion(queryNode, referenceNode->Left(),
          leftDistance, neighbors, distances);
      ComputeDualNeighborsRecursion(queryNode, referenceNode->Right(),
          rightDistance, neighbors, distances);
    }
    else
    {
      ComputeDualNeighborsRecursion(queryNode, referenceNode->Right(),
          rightDistance, neighbors, distances);
      ComputeDualNeighborsRecursion(queryNode, referenceNode->Left(),
          leftDistance, neighbors, distances);
    }
    return;
  }

  if (referenceNode->IsLeaf())
  {
    // We must descend down the query node to get to a leaf.
    double leftDistance = SortPolicy::BestNodeToNodeDistance(
        queryNode->Left(), referenceNode);
    double rightDistance = SortPolicy::BestNodeToNodeDistance(
        queryNode->Right(), referenceNode);

    ComputeDualNeighborsRecursion(queryNode->Left(), referenceNode,
        leftDistance, neighbors, distances);
    ComputeDualNeighborsRecursion(queryNode->Right(), referenceNode,
        rightDistance, neighbors, distances);

    // We need to update the upper bound based on the new upper bounds of the
    // children.
    double leftBound = queryNode->Left()->Stat().Bound();
    double rightBound = queryNode->Right()->Stat().Bound();

    if (SortPolicy::IsBetter(leftBound, rightBound))
      queryNode->Stat().Bound() = rightBound;
    else
      queryNode->Stat().Bound() = leftBound;

    return;
  }

  // Neither side is a leaf; so we recurse on all combinations of both.  The
  // calculations are ordered by distance.
  double leftDistance = SortPolicy::BestNodeToNodeDistance(queryNode->Left(),
      referenceNode->Left());
  double rightDistance = SortPolicy::BestNodeToNodeDistance(queryNode->Left(),
      referenceNode->Right());

  // Recurse on queryNode->left() first.
  if (SortPolicy::IsBetter(leftDistance, rightDistance))
  {
    ComputeDualNeighborsRecursion(queryNode->Left(), referenceNode->Left(),
        leftDistance, neighbors, distances);
    ComputeDualNeighborsRecursion(queryNode->Left(), referenceNode->Right(),
        rightDistance, neighbors, distances);
  }
  else
  {
    ComputeDualNeighborsRecursion(queryNode->Left(), referenceNode->Right(),
        rightDistance, neighbors, distances);
    ComputeDualNeighborsRecursion(queryNode->Left(), referenceNode->Left(),
        leftDistance, neighbors, distances);
  }

  leftDistance = SortPolicy::BestNodeToNodeDistance(queryNode->Right(),
      referenceNode->Left());
  rightDistance = SortPolicy::BestNodeToNodeDistance(queryNode->Right(),
      referenceNode->Right());

  // Now recurse on queryNode->right().
  if (SortPolicy::IsBetter(leftDistance, rightDistance))
  {
    ComputeDualNeighborsRecursion(queryNode->Right(), referenceNode->Left(),
        leftDistance, neighbors, distances);
    ComputeDualNeighborsRecursion(queryNode->Right(), referenceNode->Right(),
        rightDistance, neighbors, distances);
  }
  else
  {
    ComputeDualNeighborsRecursion(queryNode->Right(), referenceNode->Right(),
        rightDistance, neighbors, distances);
    ComputeDualNeighborsRecursion(queryNode->Right(), referenceNode->Left(),
        leftDistance, neighbors, distances);
  }

  // Update the upper bound as above
  double leftBound = queryNode->Left()->Stat().Bound();
  double rightBound = queryNode->Right()->Stat().Bound();

  if (SortPolicy::IsBetter(leftBound, rightBound))
    queryNode->Stat().Bound() = rightBound;
  else
    queryNode->Stat().Bound() = leftBound;

} // ComputeDualNeighborsRecursion()

template<typename SortPolicy, typename MetricType, typename TreeType>
template<typename VecType>
void NeighborSearch<SortPolicy, MetricType, TreeType>::
ComputeSingleNeighborsRecursion(const size_t pointId,
                                const VecType& point,
                                TreeType* referenceNode,
                                double& bestDistSoFar,
                                arma::Mat<size_t>& neighbors,
                                arma::mat& distances)
{
  if (referenceNode->IsLeaf())
  {
    // Base case: reference node is a leaf.
    for (size_t referenceIndex = referenceNode->Begin();
        referenceIndex < referenceNode->End(); referenceIndex++)
    {
      // Confirm that points do not identify themselves as neighbors
      // in the monochromatic case
      if (!(referenceSet.memptr() == querySet.memptr() &&
            referenceIndex == pointId))
      {
        arma::vec referencePoint = referenceSet.unsafe_col(referenceIndex);

        double distance = metric.Evaluate(point, referencePoint);

        // If the reference point is better than any of the current candidates,
        // insert it into the list correctly.
        arma::vec queryDist = distances.unsafe_col(pointId);
        size_t insertPosition = SortPolicy::SortDistance(queryDist, distance);

        if (insertPosition != (size_t() - 1))
          InsertNeighbor(pointId, insertPosition, referenceIndex, distance,
              neighbors, distances);
      }
    } // for referenceIndex

    bestDistSoFar = distances(distances.n_rows - 1, pointId);
  }
  else
  {
    // We'll order the computation by distance.
    double leftDistance = SortPolicy::BestPointToNodeDistance(point,
        referenceNode->Left());
    double rightDistance = SortPolicy::BestPointToNodeDistance(point,
        referenceNode->Right());

    // Recurse in the best direction first.
    if (SortPolicy::IsBetter(leftDistance, rightDistance))
    {
      if (SortPolicy::IsBetter(bestDistSoFar, leftDistance))
        numberOfPrunes++; // Prune; no possibility of finding a better point.
      else
        ComputeSingleNeighborsRecursion(pointId, point, referenceNode->Left(),
            bestDistSoFar, neighbors, distances);

      if (SortPolicy::IsBetter(bestDistSoFar, rightDistance))
        numberOfPrunes++; // Prune; no possibility of finding a better point.
      else
        ComputeSingleNeighborsRecursion(pointId, point, referenceNode->Right(),
            bestDistSoFar, neighbors, distances);

    }
    else
    {
      if (SortPolicy::IsBetter(bestDistSoFar, rightDistance))
        numberOfPrunes++; // Prune; no possibility of finding a better point.
      else
        ComputeSingleNeighborsRecursion(pointId, point, referenceNode->Right(),
            bestDistSoFar, neighbors, distances);

      if (SortPolicy::IsBetter(bestDistSoFar, leftDistance))
        numberOfPrunes++; // Prune; no possibility of finding a better point.
      else
        ComputeSingleNeighborsRecursion(pointId, point, referenceNode->Left(),
            bestDistSoFar, neighbors, distances);
    }
  }
}

/**
 * Helper function to insert a point into the neighbors and distances matrices.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param pos Position in list to insert into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
void NeighborSearch<SortPolicy, MetricType, TreeType>::InsertNeighbor(
    const size_t queryIndex,
    const size_t pos,
    const size_t neighbor,
    const double distance,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
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

#endif
