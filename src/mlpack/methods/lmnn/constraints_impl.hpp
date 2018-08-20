/**
 * @file constraints_impl.hpp
 * @author Manish Kumar
 *
 * Implementation of the Constraints class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_CONSTRAINTS_IMPL_HPP
#define MLPACK_METHODS_LMNN_CONSTRAINTS_IMPL_HPP

// In case it hasn't been included already.
#include "constraints.hpp"

namespace mlpack {
namespace lmnn {

template<typename MetricType>
Constraints<MetricType>::Constraints(
    const arma::mat& /* dataset */,
    const arma::Row<size_t>& labels,
    const size_t k) :
    k(k),
    tree(NULL),
    precalculated(false),
    runFirstSearch(false)
{
  // Ensure a valid k is passed.
  size_t minCount = arma::min(arma::histc(labels, arma::unique(labels)));

  if (minCount < k + 1)
  {
    Log::Fatal << "Constraints::Constraints(): One of the class contains only "
        << minCount << " instances, but value of k is " << k << "  "
        << "(k should be < " << minCount << ")!" << std::endl;
  }
}

template<typename MetricType>
Constraints<MetricType>::~Constraints()
{
  delete tree;
}

template<typename MetricType>
inline void Constraints<MetricType>::ReorderResults(
    const arma::mat& distances,
    arma::Mat<size_t>& neighbors,
    const arma::vec& norms)
{
  // Shortcut...
  if (neighbors.n_rows == 1)
    return;

  // Just a simple loop over the results---we want to make sure that the
  // largest-norm point with identical distance has the last location.
  for (size_t i = 0; i < neighbors.n_cols; i++)
  {
    for (size_t start = 0; start < neighbors.n_rows - 1; start++)
    {
      size_t end = start + 1;
      while (distances(start, i) == distances(end, i) &&
          end < neighbors.n_rows)
      {
        end++;
        if (end == neighbors.n_rows)
          break;
      }

      if (start != end)
      {
        // We must sort these elements by norm.
        arma::Col<size_t> newNeighbors =
            neighbors.col(i).subvec(start, end - 1);
        arma::uvec indices = arma::conv_to<arma::uvec>::from(newNeighbors);

        arma::uvec order = arma::sort_index(norms.elem(indices));
        neighbors.col(i).subvec(start, end - 1) =
            newNeighbors.elem(order);
      }
    }
  }
}

// Helper function to set hasImpostors and hasTrueNeighbors for a tree node.
template<typename TreeType>
void SetLMNNStat(TreeType& node,
                 const arma::Row<size_t>& labels,
                 const size_t numClasses)
{
  // If we are the root, copy the dataset.
  if (node.Parent() == NULL)
    node.Stat().OrigDataset() = new arma::mat(node.Dataset());

  // Set the size of the vectors.
  node.Stat().HasImpostors().resize(numClasses, false);
  node.Stat().HasTrueNeighbors().resize(numClasses, false);

  // We first need the results of any children.
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    TreeType& child = node.Child(i);
    SetLMNNStat(child, labels, numClasses);

    for (size_t c = 0; c < numClasses; ++c)
    {
      node.Stat().HasImpostors()[c] =
          (node.Stat().HasImpostors()[c] | child.Stat().HasImpostors()[c]);
      node.Stat().HasTrueNeighbors()[c] = (node.Stat().HasTrueNeighbors()[c] |
          child.Stat().HasTrueNeighbors()[c]);
    }
  }

  // Now compute the results of any points.
  if (node.NumPoints() > 0)
  {
    arma::Col<size_t> counts(numClasses, arma::fill::zeros);
    for (size_t i = 0; i < node.NumPoints(); ++i)
      counts[labels[node.Point(i)]]++;

    // Now, with the counts, we can determine whether impostors and true
    // neighbors are present.
    for (size_t c = 0; c < numClasses; ++c)
    {
      if (counts[c] > 0) // There is at least one true neighbor present.
        node.Stat().HasTrueNeighbors()[c] = true;
      if (counts[c] < node.NumPoints()) // There must be at least one impostor.
        node.Stat().HasImpostors()[c] = true;
    }
  }
}

// Calculates k similar labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::TargetsAndImpostors(
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    const size_t neighborsK,
    const size_t impostorsK,
    const arma::vec& norms,
    arma::Mat<size_t>& neighbors,
    arma::Mat<size_t>& impostors)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // These will be returned but not used.
  arma::mat neighborDistances, impostorDistances;

  // For now let's always do dual-tree search.
  // So, build a tree on the reference data.
  Timer::Start("tree_building");
  tree = new TreeType(dataset, oldFromNew, newFromOld);
  sortedLabels.set_size(labels.n_elem);
  sortedNorms.set_size(labels.n_elem);
  for (size_t i = 0; i < labels.n_elem; ++i)
  {
    sortedLabels[newFromOld[i]] = labels[i];
    sortedNorms[newFromOld[i]] = norms[i];
  }

  // Set the statistics correctly.
  SetLMNNStat(*tree, sortedLabels, uniqueLabels.n_cols);
  Timer::Stop("tree_building");

  MetricType metric = tree->Metric(); // No way to get an lvalue...
  LMNNTargetsAndImpostorsRules<MetricType, TreeType> rules(tree->Dataset(),
      sortedLabels, oldFromNew, tree->Dataset(), sortedLabels, oldFromNew,
      neighborsK, impostorsK, uniqueLabels.n_cols, metric);

  typename TreeType::template DualTreeTraverser<
      LMNNTargetsAndImpostorsRules<MetricType, TreeType>> traverser(rules);

  // Now perform the dual-tree traversal.
  Timer::Start("computing_targets_and_impostors");
  traverser.Traverse(*tree, *tree);

  // Next, process the results.  The unmapping is done inside the rules.
  rules.GetResults(neighbors, neighborDistances, impostors, impostorDistances);
  Timer::Stop("computing_targets_and_impostors");

  // Re-order neighbors on the basis of increasing norm in case of ties among
  // distances.
  ReorderResults(neighborDistances, neighbors);
  ReorderResults(impostorDistances, impostors);

  runFirstSearch = true;
}

// Calculates k differently labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputMatrix,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const arma::vec& norms,
                                        const arma::mat& transformation,
                                        const double transformationDiff)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // Compute all the impostors.
  arma::mat distances;
  ComputeImpostors(dataset, labels, dataset, labels, norms, transformation,
      transformationDiff, outputMatrix, distances);
}

// Calculates k differently labeled nearest neighbors. The function
// writes back calculated neighbors & distances to passed matrices.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputNeighbors,
                                        arma::mat& outputDistance,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const arma::vec& norms,
                                        const arma::mat& transformation,
                                        const double transformationDiff)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // Compute all the impostors.
  ComputeImpostors(dataset, labels, dataset, labels, norms, transformation,
      transformationDiff, outputNeighbors, outputDistance);
}

// Calculates k differently labeled nearest neighbors on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputMatrix,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const arma::vec& norms,
                                        const size_t begin,
                                        const size_t batchSize,
                                        const arma::mat& transformation,
                                        const double transformationDiff)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // Compute the impostors of the batch.
  arma::mat distances;
  arma::Mat<size_t> suboutput;
  ComputeImpostors(dataset, labels, subDataset, sublabels, norms,
      transformation, suboutput, distances, transformationDiff);
  outputMatrix.cols(begin, begin + batchSize - 1) = suboutput;
}

// Calculates k differently labeled nearest neighbors & distances on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputNeighbors,
                                        arma::mat& outputDistance,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const arma::vec& norms,
                                        const size_t begin,
                                        const size_t batchSize,
                                        const arma::mat& transformation,
                                        const double transformationDiff)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // Compute the impostors of the batch.
  arma::Mat<size_t> subneighbors;
  arma::mat subdistances;
  ComputeImpostors(dataset, labels, subDataset, sublabels, norms,
      transformation, transformationDiff, subneighbors, subdistances);
  outputNeighbors.cols(begin, begin + batchSize - 1) = subneighbors;
  outputDistance.cols(begin, begin + batchSize - 1) = subdistances;
}

template<typename MetricType>
inline void Constraints<MetricType>::Precalculate(
                                         const arma::Row<size_t>& labels)
{
  // Make sure the calculation is necessary.
  if (precalculated)
    return;

  uniqueLabels = arma::unique(labels);

  indexSame.resize(uniqueLabels.n_elem);
  indexDiff.resize(uniqueLabels.n_elem);

  for (size_t i = 0; i < uniqueLabels.n_elem; i++)
  {
    // Store same and diff indices.
    indexSame[i] = arma::find(labels == uniqueLabels[i]);
    indexDiff[i] = arma::find(labels != uniqueLabels[i]);
  }

  precalculated = true;
}

// We assume the dataset held in the tree has already been stretched.
template<typename TreeType>
inline void UpdateTree(TreeType& node, const arma::mat& transformation)
{
  node.Bound().Clear();
  if (node.NumChildren() == 0)
  {
    node.Bound() |= node.Dataset().cols(node.Point(0), node.Point(0) +
        node.NumPoints() - 1);
  }

  // Recurse into the children.
  if (node.NumChildren() > 0)
  {
    UpdateTree(node.Child(0), transformation);
    UpdateTree(node.Child(1), transformation);

    node.Bound() |= node.Left()->Bound();
    node.Bound() |= node.Right()->Bound();
  }

  // Technically this is loose but it is what the BinarySpaceTree already does.
  node.FurthestDescendantDistance() = 0.5 * node.Bound().Diameter();

  if (node.NumChildren() > 0)
  {
    // Recompute the parent distance for the left and right child.
    arma::vec center, leftCenter, rightCenter;
    node.Center(center);
    node.Child(0).Center(leftCenter);
    node.Child(1).Center(rightCenter);
    const double leftParentDistance = node.Metric().Evaluate(center,
        leftCenter);
    const double rightParentDistance = node.Metric().Evaluate(center,
        rightCenter);
    node.Child(0).ParentDistance() = leftParentDistance;
    node.Child(1).ParentDistance() = rightParentDistance;
  }
}

template<typename MetricType>
void Constraints<MetricType>::UpdateTreeStat(
    TreeType& node,
    const arma::Mat<size_t>& lastNeighbors,
    const arma::mat& lastDistances,
    const double transformationDiff)
{
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    UpdateTreeStat(node.Child(i), lastNeighbors, lastDistances,
        transformationDiff);
  }

  // Now, manually compute the bound for search.  We know that the k'th
  // nearest neighbor next iteration can't be further away than this
  // iteration's distance plus a bound.
  if (!runFirstSearch)
  {
    double relaxedBound = 0.0;
    for (size_t i = 0; i < node.NumPoints(); ++i)
    {
      const size_t index = node.Point(i);
      const size_t k = lastDistances.n_rows - 1;
      const size_t neighbor = lastNeighbors(k, index);

      const double pointBound = lastDistances(k, oldFromNew[index]) +
          transformationDiff * (sortedNorms[index] +
          sortedNorms[newFromOld[neighbor]]);

      relaxedBound = std::max(relaxedBound, pointBound);
    }

    for (size_t i = 0; i < node.NumChildren(); ++i)
      relaxedBound = std::max(relaxedBound, node.Child(i).Stat().Bound());

    node.Stat().Bound() = relaxedBound;
  }
  else
  {
    node.Stat().Bound() = DBL_MAX;
  }
}

void Constraints<MetricType>::Triplets(arma::Mat<size_t>& outputMatrix,
                                       const arma::mat& dataset,
                                       const arma::Row<size_t>& labels,
                                       const arma::vec& norms)
{
  arma::Mat<size_t> impostors(k, dataset.n_cols);
  Impostors(impostors, dataset, labels, norms);

  arma::Mat<size_t> targetNeighbors(k, dataset.n_cols);;
  TargetNeighbors(targetNeighbors, dataset, labels, norms);

  // TODO: what else went here ?
}

// Note the inputs here can just be the reference set.
template<typename MetricType>
void Constraints<MetricType>::ComputeImpostors(
    const arma::mat& referenceSet,
    const arma::Row<size_t>& /* referenceLabels */,
    const arma::mat& querySet,
    const arma::Row<size_t>& queryLabels,
    const arma::vec& norms,
    const arma::mat& transformation,
    const double transformationDiff,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
{
  // Hopefully these are already filled with last iteration's distances.
  arma::Mat<size_t> lastNeighbors = std::move(neighbors);
  arma::mat lastDistances = std::move(distances);

  // Handle the SGD case differently, where the query set is not equal to the
  // reference set.
  if (querySet.n_cols != referenceSet.n_cols)
  {
    // We'll do single-tree search instead.
    // TODO: convert from dual-tree.
    std::vector<size_t> queryOldFromNew, queryNewFromOld;
    TreeType queryTree(querySet, queryOldFromNew, queryNewFromOld);
    arma::Row<size_t> sortedQueryLabels(queryLabels.n_elem);
    for (size_t i = 0; i < queryLabels.n_elem; ++i)
      sortedQueryLabels[queryNewFromOld[i]] = queryLabels[i];

    MetricType metric = tree->Metric(); // No way to get an lvalue...
    LMNNImpostorsRules<MetricType, TreeType> rules(tree->Dataset(),
        sortedLabels, oldFromNew, queryTree.Dataset(), sortedQueryLabels,
        queryOldFromNew, k, uniqueLabels.n_cols, metric);

    typename TreeType::template DualTreeTraverser<LMNNImpostorsRules<MetricType,
        TreeType>> traverser(rules);

    // Now perform the dual-tree traversal.
    Timer::Start("computing_impostors");
    traverser.Traverse(queryTree, *tree);

    // Next, process the results.  The unmapping is done inside the rules.
    rules.GetResults(neighbors, distances);

    Timer::Stop("computing_impostors");
  }
  else
  {
    // We'll do dual-tree search on all points.
    // First we need to update the tree.  Start by stretching the dataset.
    Timer::Start("tree_stretch_dataset");
    tree->Dataset() = transformation * (*tree->Stat().OrigDataset());
    Timer::Stop("tree_stretch_dataset");
    Timer::Start("tree_update");
    UpdateTree(*tree, transformation);
    Timer::Stop("tree_update");

    // Now that the tree is ready, we have to reset the statistics for search.
    UpdateTreeStat(*tree, lastNeighbors, lastDistances, transformationDiff);
    runFirstSearch = false;

    // Now we are ready to search!
    MetricType metric = tree->Metric(); // No way to get an lvalue...
    LMNNImpostorsRules<MetricType, TreeType> rules(tree->Dataset(),
        sortedLabels, oldFromNew, tree->Dataset(), sortedLabels, oldFromNew, k,
        uniqueLabels.n_cols, metric);

    typename TreeType::template DualTreeTraverser<LMNNImpostorsRules<MetricType,
        TreeType>> traverser(rules);

    // Now perform the dual-tree traversal.
    Timer::Start("computing_impostors");
    traverser.Traverse(*tree, *tree);

    // Next, process the results.  The unmapping is done inside the rules.
    rules.GetResults(neighbors, distances);
    Timer::Stop("computing_impostors");
  }

  ReorderResults(neighbors, distances, norms);
}

} // namespace lmnn
} // namespace mlpack

#endif
