/**
 * @file constraints_impl.h
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
#include "lmnn_stat.hpp"

namespace mlpack {
namespace lmnn {

template<typename MetricType>
Constraints<MetricType>::Constraints(
    const arma::mat& /* dataset */,
    const arma::Row<size_t>& labels,
    const size_t k) :
    k(k),
    precalculated(false)
{
  // Ensure a valid k is passed.
  size_t minCount = arma::min(arma::histc(labels, arma::unique(labels)));

  if (minCount < k)
  {
    Log::Fatal << "Constraints::Constraints(): One of the class contains only "
        << minCount << " instances, but value of k is " << k << "  "
        << "(k should be < " << minCount << ")!" << std::endl;
  }
}

// Calculates k similar labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::TargetsAndImpostors(
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    const size_t neighborsK,
    const size_t impostorsK,
    arma::Mat<size_t>& neighbors,
    arma::Mat<size_t>& impostors)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  typedef tree::KDTree<MetricType, LMNNStat, arma::mat> TreeType;

  // These will be returned but not used.
  arma::mat neighborDistances, impostorDistances;

  // For now let's always do dual-tree search.
  // So, build a tree on the reference data.
  Timer::Start("tree_building");
  std::vector<size_t> oldFromNew, newFromOld;
  TreeType tree(dataset, oldFromNew, newFromOld);
  arma::Row<size_t> sortedLabels(labels.n_elem);
  for (size_t i = 0; i < labels.n_elem; ++i)
    sortedLabels[newFromOld[i]] = labels[i];

  // Set the statistics correctly.
  SetLMNNStat(tree, sortedLabels, uniqueLabels.n_cols);
  Timer::Stop("tree_building");

  MetricType metric = tree.Metric(); // No way to get an lvalue...
  LMNNTargetsAndImpostorsRules<MetricType, TreeType> rules(tree.Dataset(),
      sortedLabels, oldFromNew, tree.Dataset(), sortedLabels, oldFromNew,
      neighborsK, impostorsK, uniqueLabels.n_cols, metric);

  typename TreeType::template DualTreeTraverser<
      LMNNTargetsAndImpostorsRules<MetricType, TreeType>> traverser(rules);

  // Now perform the dual-tree traversal.
  Timer::Start("computing_targets_and_impostors");
  traverser.Traverse(tree, tree);

  // Next, process the results.  The unmapping is done inside the rules.
  rules.GetResults(neighbors, neighborDistances, impostors, impostorDistances);
  Timer::Stop("computing_targets_and_impostors");
}

// Calculates k differently labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputMatrix,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // Compute all the impostors.
  arma::mat distances;
  ComputeImpostors(dataset, labels, dataset, labels, outputMatrix, distances);
}

// Calculates k differently labeled nearest neighbors. The function
// writes back calculated neighbors & distances to passed matrices.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputNeighbors,
                                        arma::mat& outputDistance,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // Compute all the impostors.
  ComputeImpostors(dataset, labels, dataset, labels, outputNeighbors,
      outputDistance);
}

// Calculates k differently labeled nearest neighbors on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputMatrix,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const size_t begin,
                                        const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // Compute the impostors of the batch.
  arma::mat distances;
  arma::Mat<size_t> suboutput;
  ComputeImpostors(dataset, labels, subDataset, sublabels, suboutput,
      distances);
  outputMatrix.cols(begin, begin + batchSize - 1) = suboutput;
}

// Calculates k differently labeled nearest neighbors & distances on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputNeighbors,
                                        arma::mat& outputDistance,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const size_t begin,
                                        const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // Compute the impostors of the batch.
  arma::Mat<size_t> subneighbors;
  arma::mat subdistances;
  ComputeImpostors(dataset, labels, subDataset, sublabels, subneighbors,
      subdistances);
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

// Helper function to set hasImpostors and hasTrueNeighbors for a tree node.
template<typename TreeType>
void SetLMNNStat(TreeType& node,
                 const arma::Row<size_t>& labels,
                 const size_t numClasses)
{
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

// Note the inputs here can just be the reference set.
template<typename MetricType>
void Constraints<MetricType>::ComputeImpostors(
    const arma::mat& referenceSet,
    const arma::Row<size_t>& referenceLabels,
    const arma::mat& querySet,
    const arma::Row<size_t>& queryLabels,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances) const
{
  typedef tree::KDTree<MetricType, LMNNStat, arma::mat> TreeType;

  // For now let's always do dual-tree search.
  // So, build a tree on the reference data.
  Timer::Start("tree_building");
  std::vector<size_t> oldFromNew, newFromOld;
  TreeType tree(referenceSet, oldFromNew, newFromOld);
  arma::Row<size_t> sortedRefLabels(referenceLabels.n_elem);
  for (size_t i = 0; i < referenceLabels.n_elem; ++i)
    sortedRefLabels[newFromOld[i]] = referenceLabels[i];

  // Set the statistics correctly.
  SetLMNNStat(tree, sortedRefLabels, uniqueLabels.n_cols);

  // Should we build a query tree?
  TreeType* queryTree;
  arma::Row<size_t>* sortedQueryLabels;
  std::vector<size_t>* queryOldFromNew;
  std::vector<size_t>* queryNewFromOld;
  if (&querySet != &referenceSet)
  {
    queryOldFromNew = new std::vector<size_t>();
    queryNewFromOld = new std::vector<size_t>();

    queryTree = new TreeType(querySet, *queryOldFromNew,
        *queryNewFromOld);
    sortedQueryLabels = new arma::Row<size_t>(queryLabels.n_elem);
    for (size_t i = 0; i < queryLabels.n_elem; ++i)
      (*sortedQueryLabels)[(*queryNewFromOld)[i]] = queryLabels[i];
  }
  else
  {
    queryOldFromNew = &oldFromNew;
    queryNewFromOld = &newFromOld;

    queryTree = &tree;
    sortedQueryLabels = &sortedRefLabels;
  }
  Timer::Stop("tree_building");

  MetricType metric = tree.Metric(); // No way to get an lvalue...
  LMNNImpostorsRules<MetricType, TreeType> rules(tree.Dataset(),
      sortedRefLabels, oldFromNew, queryTree->Dataset(), *sortedQueryLabels,
      *queryOldFromNew, k, uniqueLabels.n_cols, metric);

  typename TreeType::template DualTreeTraverser<LMNNImpostorsRules<MetricType,
      TreeType>> traverser(rules);

  // Now perform the dual-tree traversal.
  Timer::Start("computing_impostors");
  traverser.Traverse(*queryTree, tree);

  // Next, process the results.  The unmapping is done inside the rules.
  rules.GetResults(neighbors, distances);

  Timer::Stop("computing_impostors");

  if (&querySet != &referenceSet)
  {
    delete queryOldFromNew;
    delete queryNewFromOld;

    delete queryTree;
    delete sortedQueryLabels;
  }
}

} // namespace lmnn
} // namespace mlpack

#endif
