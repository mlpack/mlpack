/**
 * @file methods/emst/dtb_impl.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Implementation of DTB.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_EMST_DTB_IMPL_HPP
#define MLPACK_METHODS_EMST_DTB_IMPL_HPP

#include "dtb_rules.hpp"

namespace mlpack {

/**
 * Takes in a reference to the data set.  Copies the data, builds the tree,
 * and initializes all of the member variables.
 */
template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType> class TreeType>
DualTreeBoruvka<DistanceType, MatType, TreeType>::DualTreeBoruvka(
    const MatType& dataset,
    const bool naive,
    const DistanceType distance) :
    tree(naive ? NULL : BuildTree<Tree>(dataset, oldFromNew)),
    data(naive ? dataset : tree->Dataset()),
    ownTree(!naive),
    naive(naive),
    connections(dataset.n_cols),
    totalDist(0.0),
    distance(distance)
{
  edges.reserve(data.n_cols - 1); // Set size.

  neighborsInComponent.set_size(data.n_cols);
  neighborsOutComponent.set_size(data.n_cols);
  neighborsDistances.set_size(data.n_cols);
  neighborsDistances.fill(DBL_MAX);
}

template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType> class TreeType>
DualTreeBoruvka<DistanceType, MatType, TreeType>::DualTreeBoruvka(
    Tree* tree,
    const DistanceType distance) :
    tree(tree),
    data(tree->Dataset()),
    ownTree(false),
    naive(false),
    connections(data.n_cols),
    totalDist(0.0),
    distance(distance)
{
  edges.reserve(data.n_cols - 1); // Fill with EdgePairs.

  neighborsInComponent.set_size(data.n_cols);
  neighborsOutComponent.set_size(data.n_cols);
  neighborsDistances.set_size(data.n_cols);
  neighborsDistances.fill(DBL_MAX);
}

template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType> class TreeType>
DualTreeBoruvka<DistanceType, MatType, TreeType>::~DualTreeBoruvka()
{
  if (ownTree)
    delete tree;
}

/**
 * Iteratively find the nearest neighbor of each component until the MST is
 * complete.
 */
template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType> class TreeType>
void DualTreeBoruvka<DistanceType, MatType, TreeType>::ComputeMST(
    arma::mat& results)
{
  totalDist = 0; // Reset distance.

  using RuleType = DTBRules<DistanceType, Tree>;
  RuleType rules(data, connections, neighborsDistances, neighborsInComponent,
                 neighborsOutComponent, distance);
  while (edges.size() < (data.n_cols - 1))
  {
    if (naive)
    {
      // Full O(N^2) traversal.
      for (size_t i = 0; i < data.n_cols; ++i)
        for (size_t j = 0; j < data.n_cols; ++j)
          rules.BaseCase(i, j);
    }
    else
    {
      typename Tree::template DualTreeTraverser<RuleType> traverser(rules);
      traverser.Traverse(*tree, *tree);
    }

    AddAllEdges();

    Cleanup();

    Log::Info << edges.size() << " edges found so far." << std::endl;
    if (!naive)
    {
      Log::Info << rules.BaseCases() << " cumulative base cases." << std::endl;
      Log::Info << rules.Scores() << " cumulative node combinations scored."
          << std::endl;
    }
  }

  EmitResults(results);

  Log::Info << "Total spanning tree length: " << totalDist << std::endl;
}

/**
 * Adds a single edge to the edge list
 */
template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType> class TreeType>
void DualTreeBoruvka<DistanceType, MatType, TreeType>::AddEdge(
    const size_t e1,
    const size_t e2,
    const double distance)
{
  Log::Assert((distance >= 0.0),
      "DualTreeBoruvka::AddEdge(): distance cannot be negative.");

  if (e1 < e2)
    edges.push_back(EdgePair(e1, e2, distance));
  else
    edges.push_back(EdgePair(e2, e1, distance));
}

/**
 * Adds all the edges found in one iteration to the list of neighbors.
 */
template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType> class TreeType>
void DualTreeBoruvka<DistanceType, MatType, TreeType>::AddAllEdges()
{
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    size_t component = connections.Find(i);
    size_t inEdge = neighborsInComponent[component];
    size_t outEdge = neighborsOutComponent[component];
    if (connections.Find(inEdge) != connections.Find(outEdge))
    {
      // totalDist = totalDist + dist;
      // changed to make this agree with the cover tree code
      totalDist += neighborsDistances[component];
      AddEdge(inEdge, outEdge, neighborsDistances[component]);
      connections.Union(inEdge, outEdge);
    }
  }
}

/**
 * Unpermute the edge list (if necessary) and output it to results.
 */
template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType> class TreeType>
void DualTreeBoruvka<DistanceType, MatType, TreeType>::EmitResults(
    arma::mat& results)
{
  // Sort the edges.
  std::sort(edges.begin(), edges.end(), SortFun);

  Log::Assert(edges.size() == data.n_cols - 1);
  results.set_size(3, edges.size());

  // Need to unpermute the point labels.
  if (!naive && ownTree && TreeTraits<Tree>::RearrangesDataset)
  {
    for (size_t i = 0; i < (data.n_cols - 1); ++i)
    {
      // Make sure the edge list stores the smaller index first to
      // make checking correctness easier
      size_t ind1 = oldFromNew[edges[i].Lesser()];
      size_t ind2 = oldFromNew[edges[i].Greater()];

      if (ind1 < ind2)
      {
        edges[i].Lesser() = ind1;
        edges[i].Greater() = ind2;
      }
      else
      {
        edges[i].Lesser() = ind2;
        edges[i].Greater() = ind1;
      }

      results(0, i) = edges[i].Lesser();
      results(1, i) = edges[i].Greater();
      results(2, i) = edges[i].Distance();
    }
  }
  else
  {
    for (size_t i = 0; i < edges.size(); ++i)
    {
      results(0, i) = edges[i].Lesser();
      results(1, i) = edges[i].Greater();
      results(2, i) = edges[i].Distance();
    }
  }
}

/**
 * This function resets the values in the nodes of the tree nearest neighbor
 * distance and checks for fully connected nodes.
 */
template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType> class TreeType>
void DualTreeBoruvka<DistanceType, MatType, TreeType>::CleanupHelper(Tree* tree)
{
  // Reset the statistic information.
  tree->Stat().MaxNeighborDistance() = DBL_MAX;
  tree->Stat().MinNeighborDistance() = DBL_MAX;
  tree->Stat().Bound() = DBL_MAX;

  // Recurse into all children.
  for (size_t i = 0; i < tree->NumChildren(); ++i)
    CleanupHelper(&tree->Child(i));

  // Get the component of the first child or point.  Then we will check to see
  // if all other components of children and points are the same.
  const int component = (tree->NumChildren() != 0) ?
      tree->Child(0).Stat().ComponentMembership() :
      connections.Find(tree->Point(0));

  // Check components of children.
  for (size_t i = 0; i < tree->NumChildren(); ++i)
    if (tree->Child(i).Stat().ComponentMembership() != component)
      return;

  // Check components of points.
  for (size_t i = 0; i < tree->NumPoints(); ++i)
    if (connections.Find(tree->Point(i)) != size_t(component))
      return;

  // If we made it this far, all components are the same.
  tree->Stat().ComponentMembership() = component;
}

/**
 * The values stored in the tree must be reset on each iteration.
 */
template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType> class TreeType>
void DualTreeBoruvka<DistanceType, MatType, TreeType>::Cleanup()
{
  for (size_t i = 0; i < data.n_cols; ++i)
    neighborsDistances[i] = DBL_MAX;

  if (!naive)
    CleanupHelper(tree);
}

} // namespace mlpack

#endif
