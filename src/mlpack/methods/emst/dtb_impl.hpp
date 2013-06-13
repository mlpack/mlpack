/**
 * @file dtb_impl.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Implementation of DTB.
 *
 * This file is part of MLPACK 1.0.6.
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

#ifndef __MLPACK_METHODS_EMST_DTB_IMPL_HPP
#define __MLPACK_METHODS_EMST_DTB_IMPL_HPP

#include "dtb_rules.hpp"

namespace mlpack {
namespace emst {

// DTBStat

/**
 * A generic initializer.
 */
DTBStat::DTBStat() : maxNeighborDistance(DBL_MAX), componentMembership(-1)
{
  // Nothing to do.
}

/**
 * An initializer for leaves.
 */
template<typename TreeType>
DTBStat::DTBStat(const TreeType& node) :
    maxNeighborDistance(DBL_MAX),
    componentMembership(((node.NumPoints() == 1) && (node.NumChildren() == 0)) ?
        node.Point(0) : -1)
{
  // Nothing to do.
}

// DualTreeBoruvka

/**
 * Takes in a reference to the data set.  Copies the data, builds the tree,
 * and initializes all of the member variables.
 */
template<typename MetricType, typename TreeType>
DualTreeBoruvka<MetricType, TreeType>::DualTreeBoruvka(
    const typename TreeType::Mat& dataset,
    const bool naive,
    const size_t leafSize,
    const MetricType metric) :
    dataCopy(dataset),
    data(dataCopy), // The reference points to our copy of the data.
    ownTree(true),
    naive(naive),
    connections(data.n_cols),
    totalDist(0.0),
    metric(metric)
{
  Timer::Start("emst/tree_building");

  if (!naive)
  {
    // Default leaf size is 1; this gives the best pruning, empirically.  Use
    // leaf_size = 1 unless space is a big concern.
    tree = new TreeType(data, oldFromNew, leafSize);
  }
  else
  {
    // Naive tree holds all data in one leaf.
    tree = new TreeType(data, oldFromNew, data.n_cols);
  }

  Timer::Stop("emst/tree_building");

  edges.reserve(data.n_cols - 1); // Set size.

  neighborsInComponent.set_size(data.n_cols);
  neighborsOutComponent.set_size(data.n_cols);
  neighborsDistances.set_size(data.n_cols);
  neighborsDistances.fill(DBL_MAX);
} // Constructor

template<typename MetricType, typename TreeType>
DualTreeBoruvka<MetricType, TreeType>::DualTreeBoruvka(
    TreeType* tree,
    const typename TreeType::Mat& dataset,
    const MetricType metric) :
    data(dataset),
    tree(tree),
    ownTree(true),
    naive(false),
    connections(data.n_cols),
    totalDist(0.0),
    metric(metric)
{
  edges.reserve(data.n_cols - 1); // fill with EdgePairs

  neighborsInComponent.set_size(data.n_cols);
  neighborsOutComponent.set_size(data.n_cols);
  neighborsDistances.set_size(data.n_cols);
  neighborsDistances.fill(DBL_MAX);
}

template<typename MetricType, typename TreeType>
DualTreeBoruvka<MetricType, TreeType>::~DualTreeBoruvka()
{
  if (ownTree)
    delete tree;
}

/**
 * Iteratively find the nearest neighbor of each component until the MST is
 * complete.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::ComputeMST(arma::mat& results)
{
  Timer::Start("emst/mst_computation");

  totalDist = 0; // Reset distance.

  typedef DTBRules<MetricType, TreeType> RuleType;
  RuleType rules(data, connections, neighborsDistances, neighborsInComponent,
                 neighborsOutComponent, metric);

  while (edges.size() < (data.n_cols - 1))
  {

    typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*tree, *tree);

    AddAllEdges();

    Cleanup();

    Log::Info << edges.size() << " edges found so far.\n";
  }

  Timer::Stop("emst/mst_computation");

  EmitResults(results);

  Log::Info << "Total squared length: " << totalDist << std::endl;
} // ComputeMST

/**
 * Adds a single edge to the edge list
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::AddEdge(const size_t e1,
                                        const size_t e2,
                                        const double distance)
{
  Log::Assert((distance >= 0.0),
      "DualTreeBoruvka::AddEdge(): distance cannot be negative.");

  if (e1 < e2)
    edges.push_back(EdgePair(e1, e2, distance));
  else
    edges.push_back(EdgePair(e2, e1, distance));
} // AddEdge

/**
 * Adds all the edges found in one iteration to the list of neighbors.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::AddAllEdges()
{
  for (size_t i = 0; i < data.n_cols; i++)
  {
    size_t component = connections.Find(i);
    size_t inEdge = neighborsInComponent[component];
    size_t outEdge = neighborsOutComponent[component];
    if (connections.Find(inEdge) != connections.Find(outEdge))
    {
      //totalDist = totalDist + dist;
      // changed to make this agree with the cover tree code
      totalDist += sqrt(neighborsDistances[component]);
      AddEdge(inEdge, outEdge, neighborsDistances[component]);
      connections.Union(inEdge, outEdge);
    }
  }
} // AddAllEdges

/**
 * Unpermute the edge list (if necessary) and output it to results.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::EmitResults(arma::mat& results)
{
  // Sort the edges.
  std::sort(edges.begin(), edges.end(), SortFun);

  Log::Assert(edges.size() == data.n_cols - 1);
  results.set_size(3, edges.size());

  // Need to unpermute the point labels.
  if (!naive && ownTree)
  {
    for (size_t i = 0; i < (data.n_cols - 1); i++)
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
      results(2, i) = sqrt(edges[i].Distance());
    }
  }
  else
  {
    for (size_t i = 0; i < edges.size(); i++)
    {
      results(0, i) = edges[i].Lesser();
      results(1, i) = edges[i].Greater();
      results(2, i) = sqrt(edges[i].Distance());
    }
  }
} // EmitResults

/**
 * This function resets the values in the nodes of the tree nearest neighbor
 * distance and checks for fully connected nodes.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::CleanupHelper(TreeType* tree)
{
  tree->Stat().MaxNeighborDistance() = DBL_MAX;

  if (!tree->IsLeaf())
  {
    CleanupHelper(tree->Left());
    CleanupHelper(tree->Right());

    if ((tree->Left()->Stat().ComponentMembership() >= 0)
        && (tree->Left()->Stat().ComponentMembership() ==
            tree->Right()->Stat().ComponentMembership()))
    {
      tree->Stat().ComponentMembership() =
          tree->Left()->Stat().ComponentMembership();
    }
  }
  else
  {
    size_t newMembership = connections.Find(tree->Begin());

    for (size_t i = tree->Begin(); i < tree->End(); ++i)
    {
      if (newMembership != connections.Find(i))
      {
        newMembership = -1;
        Log::Assert(tree->Stat().ComponentMembership() < 0);
        return;
      }
    }
    tree->Stat().ComponentMembership() = newMembership;
  }
} // CleanupHelper

/**
 * The values stored in the tree must be reset on each iteration.
 */
template<typename MetricType, typename TreeType>
void DualTreeBoruvka<MetricType, TreeType>::Cleanup()
{
  for (size_t i = 0; i < data.n_cols; i++)
  {
    neighborsDistances[i] = DBL_MAX;
  }

  if (!naive)
  {
    CleanupHelper(tree);
  }
}

}; // namespace emst
}; // namespace mlpack

#endif
