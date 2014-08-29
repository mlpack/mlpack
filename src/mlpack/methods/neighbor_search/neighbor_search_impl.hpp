/**
 * @file neighbor_search_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Neighbor-Search class to perform all-nearest-neighbors on
 * two specified data sets.
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
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>

#include "neighbor_search_rules.hpp"

namespace mlpack {
namespace neighbor {

//! Call the tree constructor that does mapping.
template<typename TreeType>
TreeType* BuildTree(
    typename TreeType::Mat& dataset,
    std::vector<size_t>& oldFromNew,
    typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == true, TreeType*
    >::type = 0)
{
  return new TreeType(dataset, oldFromNew);
}

//! Call the tree constructor that does not do mapping.
template<typename TreeType>
TreeType* BuildTree(
    const typename TreeType::Mat& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == false, TreeType*
    >::type = 0)
{
  return new TreeType(dataset);
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::
NeighborSearch(const typename TreeType::Mat& referenceSetIn,
               const typename TreeType::Mat& querySetIn,
               const bool naive,
               const bool singleMode,
               const MetricType metric) :
    referenceSet(tree::TreeTraits<TreeType>::RearrangesDataset ? referenceCopy
        : referenceSetIn),
    querySet(tree::TreeTraits<TreeType>::RearrangesDataset ? queryCopy
        : querySetIn),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(!naive), // False if a tree was passed.  If naive, then no trees.
    hasQuerySet(true),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric)
{
  // C++11 will allow us to call out to other constructors so we can avoid this
  // copy/paste problem.

  // We'll time tree building, but only if we are building trees.
  Timer::Start("tree_building");

  // Copy the datasets, if they will be modified during tree building.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    referenceCopy = referenceSetIn;
    queryCopy = querySetIn;
  }

  // If not in naive mode, then we need to build trees.
  if (!naive)
  {
    // The const_cast is safe; if RearrangesDataset == false, then it'll be
    // casted back to const anyway, and if not, referenceSet points to
    // referenceCopy, which isn't const.
    referenceTree = BuildTree<TreeType>(
        const_cast<typename TreeType::Mat&>(referenceSet),
        oldFromNewReferences);

    if (!singleMode)
      queryTree = BuildTree<TreeType>(
          const_cast<typename TreeType::Mat&>(querySet), oldFromNewQueries);
  }

  // Stop the timer we started above (if we need to).
  Timer::Stop("tree_building");
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::
NeighborSearch(const typename TreeType::Mat& referenceSetIn,
               const bool naive,
               const bool singleMode,
               const MetricType metric) :
    referenceSet(tree::TreeTraits<TreeType>::RearrangesDataset ? referenceCopy
        : referenceSetIn),
    querySet(tree::TreeTraits<TreeType>::RearrangesDataset ? referenceCopy
        : referenceSetIn),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(!naive), // If naive, then we are not building any trees.
    hasQuerySet(false),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric)
{
  // We'll time tree building, but only if we are building trees.
  Timer::Start("tree_building");

  // Copy the dataset, if it will be modified during tree building.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
    referenceCopy = referenceSetIn;

  // If not in naive mode, then we may need to construct trees.
  if (!naive)
  {
    // The const_cast is safe; if RearrangesDataset == false, then it'll be
    // casted back to const anyway, and if not, referenceSet points to
    // referenceCopy, which isn't const.
    referenceTree = BuildTree<TreeType>(
        const_cast<typename TreeType::Mat&>(referenceSet),
        oldFromNewReferences);

    if (!singleMode)
      queryTree = new TreeType(*referenceTree);
  }

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
    treeOwner(false),
    hasQuerySet(true),
    naive(false),
    singleMode(singleMode),
    metric(metric)
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
    treeOwner(false),
    hasQuerySet(false), // In this case we will own a tree, if singleMode.
    naive(false),
    singleMode(singleMode),
    metric(metric)
{
  Timer::Start("tree_building");

  // The query tree cannot be the same as the reference tree.
  if (referenceTree && !singleMode)
    queryTree = new TreeType(*referenceTree);

  Timer::Stop("tree_building");
}

/**
 * The tree is the only member we may be responsible for deleting.  The others
 * will take care of themselves.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::~NeighborSearch()
{
  if (treeOwner)
  {
    if (referenceTree)
      delete referenceTree;
    if (queryTree)
      delete queryTree;
  }
  else if (!treeOwner && !hasQuerySet && !(singleMode || naive))
  {
    // We replicated the reference tree to create a query tree.
    delete queryTree;
  }
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

  // Mapping is only necessary if the tree rearranges points.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    if (treeOwner && !(singleMode && hasQuerySet))
      distancePtr = new arma::mat; // Query indices need to be mapped.

    if (treeOwner)
      neighborPtr = new arma::Mat<size_t>; // All indices need mapping.
  }

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  neighborPtr->fill(size_t() - 1);
  distancePtr->set_size(k, querySet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());

  // Create the helper object for the tree traversal.
  typedef NeighborSearchRules<SortPolicy, MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, querySet, *neighborPtr, *distancePtr, metric);

  if (naive)
  {
    // The naive brute-force traversal.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      for (size_t j = 0; j < referenceSet.n_cols; ++j)
        rules.BaseCase(i, j);
  }
  else if (singleMode)
  {
    // The search doesn't work if the root node is also a leaf node.
    // if this is the case, it is suggested that you use the naive method.
    assert(!(referenceTree->IsLeaf()));

    // Create the traverser.
    typename TreeType::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";
  }
  else // Dual-tree recursion.
  {
    // Create the traverser.
    typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*queryTree, *referenceTree);

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";
  }

  Timer::Stop("computing_neighbors");

  // Now, do we need to do mapping of indices?
  if (!treeOwner || !tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    // No mapping needed.  We are done.
    return;
  }
  else if (treeOwner && hasQuerySet && !singleMode) // Map both sets.
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
  else if (treeOwner && !hasQuerySet)
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

    // Finished with temporary matrices.
    delete neighborPtr;
    delete distancePtr;
  }
  else if (treeOwner && hasQuerySet && singleMode) // Map only references.
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

    // Finished with temporary matrix.
    delete neighborPtr;
  }
} // Search


//Return a String of the Object.
template<typename SortPolicy, typename MetricType, typename TreeType>
std::string NeighborSearch<SortPolicy, MetricType, TreeType>::ToString() const
{
  std::ostringstream convert;
  convert << "NearestNeighborSearch [" << this << "]" << std::endl;
  convert << "  Reference Set: " << referenceSet.n_rows << "x" ;
  convert <<  referenceSet.n_cols << std::endl;
  if (&referenceSet != &querySet)
    convert << "  QuerySet: " << querySet.n_rows << "x" << querySet.n_cols
        << std::endl;
  convert << "  Reference Tree: " << referenceTree << std::endl;
  if (&referenceTree != &queryTree)
    convert << "  QueryTree: " << queryTree << std::endl;
  convert << "  Tree Owner: " << treeOwner << std::endl;
  convert << "  Naive: " << naive << std::endl;
  convert << "  Metric: " << std::endl;
  convert << mlpack::util::Indent(metric.ToString(),2);
  return convert.str();
}

}; // namespace neighbor
}; // namespace mlpack

#endif
