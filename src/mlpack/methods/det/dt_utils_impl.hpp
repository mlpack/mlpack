/**
 * @file methods/det/dt_utils_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * This file implements functions to perform different tasks with the Density
 * Tree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DET_DT_UTILS_IMPL_HPP
#define MLPACK_METHODS_DET_DT_UTILS_IMPL_HPP

#include "dt_utils.hpp"
#include <mlpack/core/tree/enumerate_tree.hpp>

namespace mlpack {

template <typename MatType>
void PrintLeafMembership(DTree<MatType, int>* dtree,
                         const MatType& data,
                         const arma::Mat<size_t>& labels,
                         const size_t numClasses,
                         const std::string& leafClassMembershipFile)
{
  // Tag the leaves with numbers.
  int numLeaves = dtree->TagTree();

  arma::Mat<size_t> table(numLeaves, (numClasses + 1));
  table.zeros();

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    const typename GetColType<MatType>::type testPoint = data.unsafe_col(i);
    const int leafTag = dtree->FindBucket(testPoint);
    const size_t label = labels[i];
    table(leafTag, label) += 1;
  }

  if (leafClassMembershipFile == "")
  {
    Log::Info << "Leaf membership; row represents leaf id, column represents "
        << "class id; value represents number of points in leaf in class."
        << std::endl << table;
  }
  else
  {
    // Create a stream for the file.
    std::ofstream outfile(leafClassMembershipFile.c_str());
    if (outfile.good())
    {
      outfile << table;
      Log::Info << "Leaf membership printed to '" << leafClassMembershipFile
          << "'." << std::endl;
    }
    else
    {
      Log::Warn << "Can't open '" << leafClassMembershipFile << "' to write "
          << "leaf membership to." << std::endl;
    }
    outfile.close();
  }

  return;
}

template <typename MatType, typename TagType>
void PrintVariableImportance(const DTree<MatType, TagType>* dtree,
                             const std::string viFile)
{
  arma::vec imps;
  dtree->ComputeVariableImportance(imps);

  double max = 0.0;
  for (size_t i = 0; i < imps.n_elem; ++i)
    if (imps[i] > max)
      max = imps[i];

  Log::Info << "Maximum variable importance: " << max << "." << std::endl;

  if (viFile == "")
  {
    Log::Info << "Variable importance: " << std::endl << imps.t() << std::endl;
  }
  else
  {
    std::ofstream outfile(viFile.c_str());
    if (outfile.good())
    {
      outfile << imps;
      Log::Info << "Variable importance printed to '" << viFile << "'."
          << std::endl;
    }
    else
    {
      Log::Warn << "Can't open '" << viFile << "' to write variable importance "
          << "to." << std::endl;
    }
    outfile.close();
  }
}


// This function trains the optimal decision tree using the given number of
// folds.
template <typename MatType, typename TagType>
DTree<MatType, TagType>* Trainer(MatType& dataset,
                                 const size_t folds,
                                 const bool useVolumeReg,
                                 const size_t maxLeafSize,
                                 const size_t minLeafSize,
                                 const bool skipPruning,
                                 util::Timers& timers)
{
  // Initialize the tree.
  DTree<MatType, TagType>* dtree = new DTree<MatType, TagType>(dataset);

  timers.Start("tree_growing");
  // Prepare to grow the tree...
  arma::Col<size_t> oldFromNew(dataset.n_cols);
  for (size_t i = 0; i < oldFromNew.n_elem; ++i)
    oldFromNew[i] = i;

  // Save the dataset since it would be modified while growing the tree.
  MatType newDataset(dataset);

  // Growing the tree
  double oldAlpha = 0.0;
  double alpha = dtree->Grow(newDataset, oldFromNew, useVolumeReg, maxLeafSize,
      minLeafSize);

  timers.Stop("tree_growing");
  Log::Info << dtree->SubtreeLeaves() << " leaf nodes in the tree using full "
      << "dataset; minimum alpha: " << alpha << "." << std::endl;

  if (skipPruning)
    return dtree;

  if (folds == dataset.n_cols)
    Log::Info << "Performing leave-one-out cross validation." << std::endl;
  else
    Log::Info << "Performing " << folds << "-fold cross validation." <<
      std::endl;

  timers.Start("pruning_sequence");

  // Sequentially prune and save the alpha values and the values of c_t^2 * r_t.
  std::vector<std::pair<double, double> > prunedSequence;
  while (dtree->SubtreeLeaves() > 1)
  {
    std::pair<double, double> treeSeq(oldAlpha,
        dtree->SubtreeLeavesLogNegError());
    prunedSequence.push_back(treeSeq);
    oldAlpha = alpha;
    alpha = dtree->PruneAndUpdate(oldAlpha, dataset.n_cols, useVolumeReg);

    // Some sanity checks.  It seems that on some datasets, the error does not
    // increase as the tree is pruned but instead stays the same---hence the
    // "<=" in the final assert.
    Log::Assert((alpha < std::numeric_limits<double>::max())
                || (dtree->SubtreeLeaves() == 1));
    Log::Assert(alpha > oldAlpha);
    Log::Assert(dtree->SubtreeLeavesLogNegError() <= treeSeq.second);
  }

  std::pair<double, double> treeSeq(oldAlpha,
                                    dtree->SubtreeLeavesLogNegError());
  prunedSequence.push_back(treeSeq);

  timers.Stop("pruning_sequence");
  Log::Info << prunedSequence.size() << " trees in the sequence; maximum alpha:"
      << " " << oldAlpha << "." << std::endl;

  const MatType cvData(dataset);
  const size_t testSize = dataset.n_cols / folds;

  arma::vec regularizationConstants(prunedSequence.size());
  regularizationConstants.fill(0.0);

  timers.Start("cross_validation");
  // Go through each fold.
  #pragma omp parallel for shared(prunedSequence, regularizationConstants)
  for (size_t fold = 0; fold < (size_t) folds; fold++)
  {
    // Break up data into train and test sets.
    const size_t start = fold * testSize;
    const size_t end = std::min((size_t) (fold + 1)
                                * testSize, (size_t) cvData.n_cols);

    MatType test = cvData.cols(start, end - 1);
    MatType train(cvData.n_rows, cvData.n_cols - test.n_cols);

    if (start == 0 && end < cvData.n_cols)
    {
      train.cols(0, train.n_cols - 1) = cvData.cols(end, cvData.n_cols - 1);
    }
    else if (start > 0 && end == cvData.n_cols)
    {
      train.cols(0, train.n_cols - 1) = cvData.cols(0, start - 1);
    }
    else
    {
      train.cols(0, start - 1) = cvData.cols(0, start - 1);
      train.cols(start, train.n_cols - 1) = cvData.cols(end, cvData.n_cols - 1);
    }

    // Initialize the tree.
    DTree<MatType, TagType> cvDTree(train);

    // Getting ready to grow the tree...
    arma::Col<size_t> cvOldFromNew(train.n_cols);
    for (size_t i = 0; i < cvOldFromNew.n_elem; ++i)
      cvOldFromNew[i] = i;

    // Grow the tree.
    cvDTree.Grow(train, cvOldFromNew, useVolumeReg, maxLeafSize,
        minLeafSize);

    // Sequentially prune with all the values of available alphas and adding
    // values for test values.  Don't enter this loop if there are less than two
    // trees in the pruned sequence.
    arma::vec cvRegularizationConstants(prunedSequence.size());
    cvRegularizationConstants.fill(0.0);
    for (size_t i = 0;
         i < ((prunedSequence.size() < 2) ? 0 : prunedSequence.size() - 2); ++i)
    {
      // Compute test values for this state of the tree.
      double cvVal = 0.0;
      for (size_t j = 0; j < test.n_cols; ++j)
      {
        arma::vec testPoint = test.unsafe_col(j);
        cvVal += cvDTree.ComputeValue(testPoint);
      }

      // Update the cv regularization constant.
      cvRegularizationConstants[i] = 2.0 * cvVal / (double) cvData.n_cols;

      // Determine the new alpha value and prune accordingly.
      double cvOldAlpha = 0.5 * (prunedSequence[i + 1].first
                                 + prunedSequence[i + 2].first);
      cvDTree.PruneAndUpdate(cvOldAlpha, train.n_cols, useVolumeReg);
    }

    // Compute test values for this state of the tree.
    double cvVal = 0.0;
    for (size_t i = 0; i < test.n_cols; ++i)
    {
      typename GetColType<MatType>::type testPoint = test.unsafe_col(i);
      cvVal += cvDTree.ComputeValue(testPoint);
    }

    if (prunedSequence.size() > 2)
    {
      cvRegularizationConstants[prunedSequence.size() - 2] = 2.0 * cvVal
        / (double) cvData.n_cols;
    }

    #pragma omp critical(DTreeCVUpdate)
    regularizationConstants += cvRegularizationConstants;
  }
  timers.Stop("cross_validation");

  double optimalAlpha = -1.0;
  long double cvBestNegError = -std::numeric_limits<long double>::max();

  for (size_t i = 0; i < prunedSequence.size() - 1; ++i)
  {
    // We can no longer work in the log-space for this because we have no
    // guarantee the quantity will be positive.
    long double thisNegError = std::exp((long double) prunedSequence[i].second)
        - (long double) regularizationConstants[i];

    if (thisNegError > cvBestNegError)
    {
      cvBestNegError = thisNegError;
      optimalAlpha = prunedSequence[i].first;
    }
  }

  Log::Info << "Optimal alpha: " << optimalAlpha << "." << std::endl;

  // Re-Initialize the tree.
  delete dtree;
  dtree = new DTree<MatType, TagType>(dataset);

  // Getting ready to grow the tree...
  for (size_t i = 0; i < oldFromNew.n_elem; ++i)
    oldFromNew[i] = i;

  // Save the dataset since it would be modified while growing the tree.
  newDataset = dataset;

  // Grow the tree.
  oldAlpha = -DBL_MAX;
  alpha = dtree->Grow(newDataset,
                         oldFromNew,
                         useVolumeReg,
                         maxLeafSize,
                         minLeafSize);

  // Prune with optimal alpha.
  while ((oldAlpha < optimalAlpha) && (dtree->SubtreeLeaves() > 1))
  {
    oldAlpha = alpha;
    alpha = dtree->PruneAndUpdate(oldAlpha, newDataset.n_cols, useVolumeReg);

    // Some sanity checks.
    Log::Assert((alpha < std::numeric_limits<double>::max()) ||
        (dtree->SubtreeLeaves() == 1));
    Log::Assert(alpha > oldAlpha);
  }

  Log::Info << dtree->SubtreeLeaves() << " leaf nodes in the optimally "
      << "pruned tree; optimal alpha: " << oldAlpha << "." << std::endl;

  return dtree;
}

template<typename MatType>
PathCacher::PathCacher(PathCacher::PathFormat fmt, DTree<MatType, int>* dtree) :
    format(fmt)
{
  // Here we use TagTree()'s output to determine the
  // number of _nodes_ in the tree.
  pathCache.resize(dtree->TagTree(0, true));
  pathCache[0] = PathCacheType::value_type(-1, "");
  EnumerateTree(dtree, *this);
}

template<typename MatType>
void PathCacher::Enter(const DTree<MatType, int>* node,
                       const DTree<MatType, int>* parent)
{
  if (parent == nullptr)
    return;

  int tag = node->BucketTag();

  path.push_back(PathType::value_type(parent->Left() == node, tag));
  pathCache[tag] = PathCacheType::value_type(parent->BucketTag(),
                                             (node->SubtreeLeaves() > 1) ?
                                             "" : BuildString());
}

template<typename MatType>
void PathCacher::Leave(const DTree<MatType, int>* /* node */,
                       const DTree<MatType, int>* parent)
{
  if (parent != nullptr)
    path.pop_back();
}

std::string PathCacher::BuildString()
{
  std::string str("");
  for (PathType::iterator it = path.begin(); it != path.end(); it++)
  {
    switch (format)
    {
      case FormatLR:
        str += it->first ? "L" : "R";
        break;
      case FormatLR_ID:
        str += (it->first ? "L" : "R") + std::to_string(it->second);
        break;
      case FormatID_LR:
        str += std::to_string(it->second) + (it->first ? "L" : "R");
        break;
    }
  }

  return str;
}

int PathCacher::ParentOf(int tag) const
{
  return pathCache[tag].first;
}

const std::string& PathCacher::PathFor(int tag) const
{
  return pathCache[tag].second;
}

} // namespace mlpack

#endif // MLPACK_METHODS_DET_DT_UTILS_IMPL_HPP
