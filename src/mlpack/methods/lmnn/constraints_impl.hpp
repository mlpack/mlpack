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
    const size_t k,
    const double rebuildTolerance) :
    k(k),
    precalculated(false),
    runFirstSearch(true),
    rebuildTolerance(rebuildTolerance)
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
  for (size_t i = 0; i < trees.size(); ++i)
    delete trees[i];
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

// Calculates k similar labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::TargetsAndImpostors(
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    const size_t neighborsK,
    const arma::vec& norms,
    arma::Mat<size_t>& neighbors,
    arma::Mat<size_t>& impostors,
    arma::mat& impostorDistances)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(dataset, labels);

  // Compute targets with a search for each class.  We can use the existing KNN
  // search rules code for this.
  arma::mat neighborDistances(neighborsK, dataset.n_cols);
  neighbors.set_size(neighborsK, dataset.n_cols);
  MetricType metric = trees[0]->Metric(); // We need an instantiated object.
  for (size_t i = 0; i < trees.size(); ++i)
  {
    LMNNTargetsRules<MetricType, TreeType> rules(trees[i]->Dataset(),
        neighborsK, metric);

    typename TreeType::template DualTreeTraverser<LMNNTargetsRules<MetricType,
        TreeType>> traverser(rules);

    // Now perform the dual-tree traversal.
    Timer::Start("computing_targets_and_impostors");
    traverser.Traverse(*trees[i], *trees[i]);

    // Next, process the results.
    rules.GetResults(oldFromNews[i], neighbors, neighborDistances);
    Timer::Stop("computing_targets_and_impostors");
  }
  ReorderResults(neighborDistances, neighbors, norms);

  // Now that the targets are computed, we need to compute the impostors.
  ComputeImpostors(dataset, norms, arma::eye<arma::mat>(dataset.n_rows,
      dataset.n_rows), 0.0, false, impostors, impostorDistances);
}

// Calculates k differently labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::Impostors(
    arma::Mat<size_t>& outputMatrix,
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    const arma::vec& norms,
    const arma::mat& transformation,
    const double transformationDiff,
    const bool useImpBounds)
{
  Timer::Start("impostors");
  // Perform pre-calculation. If neccesary.
  Precalculate(dataset, labels);

  // Compute all the impostors.
  arma::mat distances;
  ComputeImpostors(dataset, norms, transformation, transformationDiff,
      useImpBounds, outputMatrix, distances);
  Timer::Stop("impostors");
}

// Calculates k differently labeled nearest neighbors. The function
// writes back calculated neighbors & distances to passed matrices.
template<typename MetricType>
void Constraints<MetricType>::Impostors(
    arma::Mat<size_t>& outputNeighbors,
    arma::mat& outputDistance,
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    const arma::vec& norms,
    const arma::mat& transformation,
    const double transformationDiff,
    const bool useImpBounds)
{
  Timer::Start("impostors");
  // Perform pre-calculation. If neccesary.
  Precalculate(dataset, labels);

  // Compute all the impostors.
  ComputeImpostors(dataset, norms, transformation, transformationDiff,
      useImpBounds, outputNeighbors, outputDistance);
  Timer::Stop("impostors");
}

// Calculates k differently labeled nearest neighbors on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(
    arma::Mat<size_t>& outputMatrix,
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    const arma::vec& /* norms */,
    const size_t begin,
    const size_t batchSize,
    const arma::mat& /* transformation */,
    const double /* transformationDiff */,
    const bool /* useImpBounds */)
{
  Timer::Start("impostors");
  // Perform pre-calculation. If neccesary.
  Precalculate(dataset, labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // Compute the impostors of the batch.
  arma::mat distances;
  arma::Mat<size_t> suboutput;
  Log::Fatal << "no" << std::endl;
  //ComputeImpostors(dataset, labels, subDataset, sublabels, norms,
  //    transformation, transformationDiff, useImpBounds, suboutput, distances);
  outputMatrix.cols(begin, begin + batchSize - 1) = suboutput;
  Timer::Stop("impostors");
}

// Calculates k differently labeled nearest neighbors & distances on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(
    arma::Mat<size_t>& outputNeighbors,
    arma::mat& outputDistance,
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    const arma::vec& /* norms */,
    const size_t begin,
    const size_t batchSize,
    const arma::mat& /* transformation */,
    const double /* transformationDiff */,
    const bool /* useImpBounds */)
{
  Timer::Start("impostors");
  // Perform pre-calculation. If neccesary.
  Precalculate(dataset, labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // Compute the impostors of the batch.
  arma::Mat<size_t> subneighbors;
  arma::mat subdistances;
  Log::Fatal << "no" << std::endl;
  //ComputeImpostors(dataset, labels, subDataset, sublabels, norms,
  //    transformation, transformationDiff, useImpBounds, subneighbors,
  //    subdistances);
  outputNeighbors.cols(begin, begin + batchSize - 1) = subneighbors;
  outputDistance.cols(begin, begin + batchSize - 1) = subdistances;
  Timer::Stop("impostors");
}

template<typename MetricType>
inline void Constraints<MetricType>::Precalculate(
    const arma::mat& dataset,
    const arma::Row<size_t>& labels)
{
  // Make sure the calculation is necessary.
  if (precalculated)
    return;

  uniqueLabels = arma::unique(labels);

  trees.resize(uniqueLabels.n_elem, NULL);
  oldFromNews.resize(uniqueLabels.n_elem);
  for (size_t i = 0; i < uniqueLabels.n_elem; i++)
  {
    arma::uvec index = arma::find(labels == uniqueLabels[i]);
    trees[i] = new TreeType(std::move(dataset.cols(index)), oldFromNews[i]);
    // Only set the original dataset for the root.
    trees[i]->Stat().OrigDataset() = new arma::mat(trees[i]->Dataset());

    // We have to make the tree's mappings into the whole dataset mapping.
    for (size_t j = 0; j < oldFromNews[i].size(); ++j)
    {
      oldFromNews[i][j] = index[oldFromNews[i][j]];
    }
  }

  lastTreeTransformation = arma::eye<arma::mat>(dataset.n_rows, dataset.n_rows);

  precalculated = true;
}

// Apply impostor bounds to filter out points or entire nodes.
template<typename TreeType>
inline void ImpBoundFilterTree(
    TreeType& node,
    const std::vector<size_t>& oldFromNew,
    const arma::vec& unsortedNorms,
    const arma::Mat<size_t>& oldImpostors,
    const arma::mat& oldDistances,
    const size_t k,
    const double transformationDiff,
    std::vector<bool>& pruned)
{
  // First, see which points we can filter.
  bool fullyFiltered = false;
  for (size_t i = 0; i < node.NumPoints(); ++i)
  {
    const size_t index = oldFromNew[node.Point(i)];
    if (transformationDiff * (2 * unsortedNorms[index] +
        unsortedNorms[oldImpostors(k - 1, index)] +
        unsortedNorms[oldImpostors(k, index)]) >
        oldDistances(k, index) - oldDistances(k - 1, index))
    {
//      pruned[node.Point(i)] = true;
    }
    else
    {
      fullyFiltered = false;
    }
  }

  // Now see if the children are filtered (if there are any).
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    ImpBoundFilterTree(node.Child(i), oldFromNew, unsortedNorms, oldImpostors,
        oldDistances, k, transformationDiff, pruned);
    fullyFiltered &= node.Child(i).Stat().Pruned();
  }

  node.Stat().Pruned() = fullyFiltered;
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
    const std::vector<size_t>& oldFromNew,
    const arma::vec& norms,
    const arma::Mat<size_t>& lastNeighbors,
    const arma::mat& lastDistances,
    const double transformationDiff,
    arma::vec& pointBounds)
{
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    UpdateTreeStat(node.Child(i), oldFromNew, norms, lastNeighbors,
        lastDistances, transformationDiff, pointBounds);
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
      const size_t oldIndex = oldFromNew[node.Point(i)];

      pointBounds[node.Point(i)] = lastDistances(k, oldIndex) +
          transformationDiff * (norms[oldIndex] + norms[neighbor]);

      relaxedBound = std::max(relaxedBound, pointBounds[node.Point(i)]);
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

template<typename MetricType>
void Constraints<MetricType>::Triplets(arma::Mat<size_t>& outputMatrix,
                                       const arma::mat& dataset,
                                       const arma::Row<size_t>& labels,
                                       const arma::vec& norms)
{
  arma::Mat<size_t> impostors;
  //Impostors(impostors, dataset, labels, norms);

  arma::Mat<size_t> targetNeighbors;
  //TargetNeighbors(targetNeighbors, dataset, labels, norms);

  // TODO: what else went here ?
}

// Note the inputs here can just be the reference set.
template<typename MetricType>
void Constraints<MetricType>::ComputeImpostors(
    const arma::mat& dataset,
    const arma::vec& norms,
    const arma::mat& transformation,
    const double transformationDiff,
    const bool useImpBounds,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
{
  // Hopefully these are already filled with last iteration's distances.
  arma::Mat<size_t> lastNeighbors = std::move(neighbors);
  arma::mat lastDistances = std::move(distances);

  neighbors.set_size(k, dataset.n_cols);
  distances.set_size(k, dataset.n_cols);

  // First we need to update the tree or rebuild it depending on the
  // transformation matrix difference.
  if (!runFirstSearch)
  {
    const double treeTransDiff = arma::norm(lastTreeTransformation -
        transformation, 2);
    if (treeTransDiff > rebuildTolerance)
    {
      Log::Info << "Rebuild trees!" << std::endl;
      Timer::Start("tree_rebuilding");
      // Rebuild the trees on the new dataset.
      for (size_t t = 0; t < trees.size(); ++t)
      {
        // Copy new points.
        arma::mat newData(trees[t]->Dataset().n_rows, trees[t]->Dataset().n_cols);
        for (size_t i = 0; i < newData.n_cols; ++i)
          newData.col(i) = dataset.col(oldFromNews[t][i]);

        std::vector<size_t> oldFromNew;

        delete trees[t];
        trees[t] = new TreeType(std::move(newData), oldFromNew);

        // Now we need to update the indices.
        std::vector<size_t> updatedOldFromNew;
        updatedOldFromNew.resize(oldFromNews[t].size());
        for (size_t i = 0; i < oldFromNews[t].size(); ++i)
          updatedOldFromNew[i] = oldFromNews[t][oldFromNew[i]];

        oldFromNews[t] = std::move(updatedOldFromNew);
      }

      // Store transformation.
      lastTreeTransformation = transformation;
      Timer::Stop("tree_rebuilding");
    }
    else
    {
      // Copy columns.
      for (size_t t = 0; t < trees.size(); ++t)
      {
        for (size_t i = 0; i < oldFromNews[t].size(); ++i)
        {
          trees[t]->Dataset().col(i) = dataset.col(oldFromNews[t][i]);
        }
        UpdateTree(*trees[t], transformation);
      }
    }
  }

  // Auxiliary variable to be used during search.
  MetricType metric = trees[0]->Metric(); // No way to get an lvalue...

  // We need to search for impostors for each combination of classes.
  for (size_t i = 0; i < uniqueLabels.n_elem; ++i)
  {
    // Before searching, we have to reset the statistics in our query tree.
    Timer::Start("tree_bound_update");
    arma::vec pointBounds(trees[i]->Dataset().n_cols);
    UpdateTreeStat(*trees[i], oldFromNews[i], norms, lastNeighbors,
        lastDistances, transformationDiff, pointBounds);
    Timer::Stop("tree_bound_update");

    // Attempt to perform impostor bound pruning.
    std::vector<bool> pruned;
    if (useImpBounds)
    {
      // Compute which points are pruned this iteration from class i.
      pruned.resize(trees[i]->Dataset().n_cols, false);
      if (!runFirstSearch)
      {
        ImpBoundFilterTree(*trees[i], oldFromNews[i], norms, lastNeighbors,
            lastDistances, lastNeighbors.n_rows - 1, transformationDiff,
            pruned);
      }
    }

    // Fill the initial candidate vector; if possible, use information from
    // previous iterations.
    std::vector<CandidateList> candidates;
    candidates.reserve(trees[i]->Dataset().n_cols);
    if (runFirstSearch)
    {
      const Candidate def = std::make_tuple(DBL_MAX, 0, size_t() - 1);
      std::vector<Candidate> vect(k, def);
      CandidateList pqueue(CandidateCmp(), std::move(vect));

      for (size_t j = 0; j < trees[i]->Dataset().n_cols; ++j)
        candidates.push_back(pqueue);
    }
    else
    {
      // Use the individually computed point bounds from UpdateTreeStat().
      for (size_t j = 0; j < trees[i]->Dataset().n_cols; ++j)
      {
        const Candidate def = std::make_tuple(std::nextafter(pointBounds[j],
            DBL_MAX), 0, size_t() - 1);
        std::vector<Candidate> vect(k, def);
        CandidateList pqueue(CandidateCmp(), std::move(vect));
        candidates.push_back(pqueue);
      }
    }

    // Now search with all other classes as the reference set.
    for (size_t j = 0; j < uniqueLabels.n_elem; ++j)
    {
      if (i == j)
        continue;

      // We need to pass in the index of the reference tree, to be used as part
      // of the output.
      if (useImpBounds)
      {
        LMNNImpostorsRules<MetricType, TreeType, true> rules(
            trees[j]->Dataset(), j, trees[i]->Dataset(), pruned, k, metric,
            candidates);

        typename TreeType::template DualTreeTraverser<
            LMNNImpostorsRules<MetricType, TreeType, true>> traverser(rules);

        Timer::Start("computing_impostors");
        traverser.Traverse(*trees[i], *trees[j]);
        Timer::Stop("computing_impostors");
      }
      else
      {
        LMNNImpostorsRules<MetricType, TreeType, false> rules(
            trees[j]->Dataset(), j, trees[i]->Dataset(), pruned, k, metric,
            candidates);

        typename TreeType::template DualTreeTraverser<
            LMNNImpostorsRules<MetricType, TreeType, false>> traverser(rules);

        Timer::Start("computing_impostors");
        traverser.Traverse(*trees[i], *trees[j]);
        Timer::Stop("computing_impostors");
      }
    }

    // Now, after all of the searches, we have to unmap each of the candidates.
    for (size_t j = 0; j < trees[i]->Dataset().n_cols; ++j)
    {
      CandidateList& pqueue = candidates[j];
      const size_t queryIndex = oldFromNews[i][j];
      for (size_t l = 1; l <= k; ++l)
      {
        const Candidate& t = pqueue.top();
        neighbors(k - l, queryIndex) =
            oldFromNews[std::get<1>(t)][std::get<2>(t)];
        distances(k - l, queryIndex) = std::get<0>(t);
        pqueue.pop();
      }
    }

    // Lastly, we have to recalculate the distances for any pruned points.
    if (useImpBounds)
    {
      Timer::Start("update_pruned_impostors");
      for (size_t j = 0; j < pruned.size(); ++j)
      {
        if (pruned[j])
        {
          const size_t index = oldFromNews[i][j];

          // Recalculate all distances.
          for (size_t k = 0; k < distances.n_rows; ++k)
          {
            neighbors(k, index) = lastNeighbors(k, index);
            distances(k, index) = metric.Evaluate(trees[i]->Dataset().col(j),
                dataset.col(neighbors(k, index)));
          }
        }
      }
      Timer::Stop("update_pruned_impostors");
    }
  }
  runFirstSearch = false;

  ReorderResults(distances, neighbors, norms);
}

} // namespace lmnn
} // namespace mlpack

#endif
