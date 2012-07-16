 /**
 * @file dtree_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * Implementations of some declared functions in
 * the Density Estimation Tree class.
 *
 */
#ifndef __MLPACK_METHODS_DET_DTREE_IMPL_HPP
#define __MLPACK_METHODS_DET_DTREE_IMPL_HPP

#include "dtree.hpp"

namespace mlpack {
namespace det {

// This function computes the log-l2-negative-error of a given node from the
// formula R(t) = log(|t|^2 / (N^2 V_t)).
template<typename eT, typename cT>
inline double DTree<eT, cT>::LogNegativeError(const size_t totalPoints) const
{
  // log(-|t|^2 / (N^2 V_t)) = log(-1) + 2 log(|t|) - 2 log(N) - log(V_t).
  return 2 * std::log((double) (end_ - start_)) -
         2 * std::log((double) totalPoints) -
         arma::accu(arma::log(maxVals - minVals));
}

// This function finds the best split with respect to the L2-error, by trying
// all possible splits.  The dataset is the full data set but the start_ and
// end_ are used to obtain the point in this node.
template<typename eT, typename cT>
bool DTree<eT, cT>::FindSplit(const arma::mat& data,
                              size_t& splitDim,
                              double& splitValue,
                              double& leftError,
                              double& rightError,
                              const size_t maxLeafSize,
                              const size_t minLeafSize) const
{
  // Ensure the dimensionality of the data is the same as the dimensionality of
  // the bounding rectangle.
  assert(data.n_rows == maxVals.n_elem);
  assert(data.n_rows == minVals.n_elem);

  const size_t points = end_ - start_;

  double minError = std::log(-error_);
  bool splitFound = false;

  // Loop through each dimension.
  for (size_t dim = 0; dim < maxVals.n_elem; dim++)
  {
    // Have to deal with REAL, INTEGER, NOMINAL data differently, so we have to
    // think of how to do that...
    const double min = minVals[dim];
    const double max = maxVals[dim];

    // If there is nothing to split in this dimension, move on.
    if (max - min == 0.0)
      continue; // Skip to next dimension.

    // Initializing all the stuff for this dimension.
    bool dimSplitFound = false;
    // Take an error estimate for this dimension.
    double minDimError = points / (max - min);
    double dimLeftError;
    double dimRightError;
    double dimSplitValue;

    // Find the log volume of all the other dimensions.
    double volumeWithoutDim = 0;
    for (size_t i = 0; i < maxVals.n_elem; ++i)
    {
      if ((maxVals[i] - minVals[i] > 0.0) && (i != dim))
      {
        volumeWithoutDim += std::log(maxVals[i] - minVals[i]);
      }
    }

    // Get the values for the dimension.
    arma::rowvec dimVec = data.row(dim).subvec(start_, end_ - 1);

    // Sort the values in ascending order.
    dimVec = arma::sort(dimVec);

    // Get ready to go through the sorted list and compute error.
    assert(dimVec.n_elem > maxLeafSize);

    // Find the best split for this dimension.  We need to figure out why
    // there are spikes if this min_leaf_size is enforced here...
    for (size_t i = minLeafSize - 1; i < dimVec.n_elem - minLeafSize; ++i)
    {
      // This makes sense for real continuous data.  This kinda corrupts the
      // data and estimation if the data is ordinal.
      const double split = (dimVec[i] + dimVec[i + 1]) / 2.0;

      if (split == dimVec[i])
        continue; // We can't split here (two points are the same).

      // Another way of picking split is using this:
      //   split = left_split;
      if ((split - min > 0.0) && (max - split > 0.0))
      {
        // Ensure that the right node will have at least the minimum number of
        // points.
        Log::Assert((points - i - 1) >= minLeafSize);

        // Now we have to see if the error will be reduced.  Simple manipulation
        // of the error function gives us the condition we must satisfy:
        //   |t_l|^2 / V_l + |t_r|^2 / V_r  >= |t|^2 / (V_l + V_r)
        // and because the volume is only dependent on the dimension we are
        // splitting, we can assume V_l is just the range of the left and V_r is
        // just the range of the right.
        double negLeftError = std::pow(i + 1, 2.0) / (split - min);
        double negRightError = std::pow(points - i - 1, 2.0) / (max - split);

        // If this is better, take it.
        if ((negLeftError + negRightError) >= minDimError)
        {
          minDimError = negLeftError + negRightError;
          dimLeftError = negLeftError;
          dimRightError = negRightError;
          dimSplitValue = split;
          dimSplitFound = true;
        }
      }
    }

    double actualMinDimError = std::log(minDimError) - 2 * std::log(data.n_cols)
        - volumeWithoutDim;

    if ((actualMinDimError > minError) && dimSplitFound)
    {
      // Calculate actual error (in logspace) by adding terms back to our
      // estimate.
      minError = actualMinDimError;
      splitDim = dim;
      splitValue = dimSplitValue;
      leftError = std::log(dimLeftError) - 2 * std::log(data.n_cols) -
          volumeWithoutDim;
      rightError = std::log(dimRightError) - 2 * std::log(data.n_cols) -
          volumeWithoutDim;
      splitFound = true;
    } // end if better split found in this dimension.
  }

  // Map out of logspace.
  minError = -std::exp(minError);
  leftError = -std::exp(leftError);
  rightError = -std::exp(rightError);

  return splitFound;
}

template<typename eT, typename cT>
size_t DTree<eT, cT>::SplitData(arma::mat& data,
                                const size_t splitDim,
                                const double splitValue,
                                arma::Col<size_t>& oldFromNew) const
{
  // Swap all columns such that any columns with value in dimension splitDim
  // less than or equal to splitValue are on the left side, and all others are
  // on the right side.  A similar sort to this is also performed in
  // BinarySpaceTree construction (its comments are more detailed).
  size_t left = start_;
  size_t right = end_ - 1;
  for (;;)
  {
    while (data(splitDim, left) <= splitValue)
      ++left;
    while (data(splitDim, right) > splitValue)
      --right;

    if (left > right)
      break;

    data.swap_cols(left, right);

    // Store the mapping from old to new.
    const size_t tmp = oldFromNew[left];
    oldFromNew[left] = oldFromNew[right];
    oldFromNew[right] = tmp;
  }

  // This now refers to the first index of the "right" side.
  return left;
}


template<typename eT, typename cT>
DTree<eT, cT>::DTree() :
    start_(0),
    end_(0),
    left_(NULL),
    right_(NULL)
{ /* Nothing to do. */ }


// Root node initializers
template<typename eT, typename cT>
DTree<eT, cT>::DTree(const arma::vec& maxVals,
                     const arma::vec& minVals,
                     const size_t totalPoints) :
    start_(0),
    end_(totalPoints),
    maxVals(maxVals),
    minVals(minVals),
    error_(-std::exp(LogNegativeError(totalPoints))),
    root_(true),
    bucket_tag_(-1),
    left_(NULL),
    right_(NULL)
{ /* Nothing to do. */ }

template<typename eT, typename cT>
DTree<eT, cT>::DTree(arma::mat& data) :
    start_(0),
    end_(data.n_cols),
    left_(NULL),
    right_(NULL)
{
  maxVals.set_size(data.n_rows);
  minVals.set_size(data.n_rows);

  // Initialize to first column; values will be overwritten if necessary.
  maxVals = data.col(0);
  minVals = data.col(0);

  // Loop over data to extract maximum and minimum values in each dimension.
  for (size_t i = 1; i < data.n_cols; ++i)
  {
    for (size_t j = 0; j < data.n_rows; ++j)
    {
      if (data(j, i) > maxVals[j])
        maxVals[j] = data(j, i);
      if (data(j, i) < minVals[j])
        minVals[j] = data(j, i);
    }
  }

  error_ = -std::exp(LogNegativeError(data.n_cols));

  bucket_tag_ = -1;
  root_ = true;
}


// Non-root node initializers
template<typename eT, typename cT>
DTree<eT, cT>::DTree(const arma::vec& maxVals,
                     const arma::vec& minVals,
                     const size_t start,
                     const size_t end,
                     const double error) :
    start_(start),
    end_(end),
    maxVals(maxVals),
    minVals(minVals),
    error_(error),
    root_(false),
    bucket_tag_(-1),
    left_(NULL),
    right_(NULL)
{ /* Nothing to do. */ }

template<typename eT, typename cT>
DTree<eT, cT>::DTree(const arma::vec& maxVals,
                     const arma::vec& minVals,
                     const size_t totalPoints,
                     const size_t start,
                     const size_t end) :
    start_(start),
    end_(end),
    maxVals(maxVals),
    minVals(minVals),
    error_(-std::exp(LogNegativeError(totalPoints))),
    root_(false),
    bucket_tag_(-1),
    left_(NULL),
    right_(NULL)
{ /* Nothing to do. */ }

template<typename eT, typename cT>
DTree<eT, cT>::~DTree()
{
  if (left_ != NULL)
    delete left_;

  if (right_ != NULL)
    delete right_;
}


// Greedily expand the tree
template<typename eT, typename cT>
double DTree<eT, cT>::Grow(arma::mat& data,
                           arma::Col<size_t>& oldFromNew,
                           const bool useVolReg,
                           const size_t maxLeafSize,
                           const size_t minLeafSize)
{
  assert(data.n_rows == maxVals.n_elem);
  assert(data.n_rows == minVals.n_elem);

  double leftG, rightG;

  // Compute points ratio.
  ratio_ = (double) (end_ - start_) / (double) oldFromNew.n_elem;

  // Compute the v_t_inv: the inverse of the volume of the node.  We use log to
  // prevent overflow.
  double logVol = 0;
  for (size_t i = 0; i < maxVals.n_elem; ++i)
    if (maxVals[i] - minVals[i] > 0.0)
      logVol += std::log(maxVals[i] - minVals[i]);

  // Check for overflow.
  assert(std::exp(logVol) > 0.0);
  v_t_inv_ = 1.0 / std::exp(logVol);

  // Check if node is large enough to split.
  if ((size_t) (end_ - start_) > maxLeafSize) {

    // Find the split.
    size_t dim;
    double splitValue;
    double leftError, rightError;
    if (FindSplit(data, dim, splitValue, leftError, rightError, maxLeafSize,
        minLeafSize))
    {
      // Move the data around for the children to have points in a node lie
      // contiguously (to increase efficiency during the training).
      const size_t splitIndex = SplitData(data, dim, splitValue, oldFromNew);

      // Make max and min vals for the children.
      arma::vec max_vals_l(maxVals);
      arma::vec max_vals_r(maxVals);
      arma::vec min_vals_l(minVals);
      arma::vec min_vals_r(minVals);

      max_vals_l[dim] = splitValue;
      min_vals_r[dim] = splitValue;

      // Store split dim and split val in the node.
      split_value_ = splitValue;
      split_dim_ = dim;

      // Recursively grow the children.
      left_ = new DTree(max_vals_l, min_vals_l, start_, splitIndex, leftError);
      right_ = new DTree(max_vals_r, min_vals_r, splitIndex, end_, rightError);

      leftG = left_->Grow(data, oldFromNew, useVolReg, maxLeafSize,
          minLeafSize);
      rightG = right_->Grow(data, oldFromNew, useVolReg, maxLeafSize,
          minLeafSize);

      // Store values of R(T~) and |T~|.
      subtree_leaves_ = left_->subtree_leaves() + right_->subtree_leaves();
      subtree_leaves_error_ = left_->subtree_leaves_error() +
          right_->subtree_leaves_error();

      // Store the subtree_leaves_v_t_inv.
      subtree_leaves_v_t_inv_ = left_->subtree_leaves_v_t_inv() +
          right_->subtree_leaves_v_t_inv();
    }
    else
    {
      // No split found so make a leaf out of it.
      subtree_leaves_ = 1;
      subtree_leaves_error_ = error_;
      subtree_leaves_v_t_inv_ = v_t_inv_;
    }
  }
  else
  {
    // We can make this a leaf node.
    assert((size_t) (end_ - start_) >= minLeafSize);
    subtree_leaves_ = 1;
    subtree_leaves_error_ = error_;
    subtree_leaves_v_t_inv_ = v_t_inv_;
  }

  // If this is a leaf, do not compute g_k(t); otherwise compute, store, and
  // propagate min(g_k(t_L), g_k(t_R), g_k(t)), unless t_L and/or t_R are
  // leaves.
  if (subtree_leaves_ == 1)
  {
    return std::numeric_limits<double>::max();
  }
  else
  {
    double g_t;
    if (useVolReg)
      g_t = (error_ - subtree_leaves_error_) /
          (subtree_leaves_v_t_inv_ - v_t_inv_);
    else
      g_t = (error_ - subtree_leaves_error_) / (subtree_leaves_ - 1);

    assert(g_t > 0.0);
    return min(g_t, min(leftG, rightG));
  }

  // We need to compute (c_t^2) * r_t for all subtree leaves; this is equal to
  // n_t ^ 2 / r_t * n ^ 2 = -error_.  Therefore the value we need is actually
  // -1.0 * subtree_leaves_error_.
}


template<typename eT, typename cT>
double DTree<eT, cT>::PruneAndUpdate(const double oldAlpha,
                                     const bool useVolReg)
{
  // Compute g_t.
  if (subtree_leaves_ == 1) // If we are a leaf...
  {
    return std::numeric_limits<double>::max();
  }
  else
  {
    // Compute g_t value for node t.
    double g_t;
    if (useVolReg)
      g_t = (error_ - subtree_leaves_error_) /
          (subtree_leaves_v_t_inv_ - v_t_inv_);
    else
      g_t = (error_ - subtree_leaves_error_) / (subtree_leaves_ - 1);

    if (g_t > oldAlpha)
    {
      // Go down the tree and update accordingly.  Traverse the children.
      double left_g = left_->PruneAndUpdate(oldAlpha, useVolReg);
      double right_g = right_->PruneAndUpdate(oldAlpha, useVolReg);

      // Update values.
      subtree_leaves_ = left_->subtree_leaves() + right_->subtree_leaves();
      subtree_leaves_error_ = left_->subtree_leaves_error() +
          right_->subtree_leaves_error();
      subtree_leaves_v_t_inv_ = left_->subtree_leaves_v_t_inv() +
          right_->subtree_leaves_v_t_inv();

      // Update g_t value.
      if (useVolReg)
        g_t = (error_ - subtree_leaves_error_)
            / (subtree_leaves_v_t_inv_ - v_t_inv_);
      else
        g_t = (error_ - subtree_leaves_error_) / (subtree_leaves_ - 1);

      assert(g_t < std::numeric_limits<double>::max());

      if (left_->subtree_leaves() == 1 && right_->subtree_leaves() == 1)
        return g_t;
      else if (left_->subtree_leaves() == 1)
        return min(g_t, right_g);
      else if (right_->subtree_leaves() == 1)
        return min(g_t, left_g);
      else
        return min(g_t, min(left_g, right_g));
    }
    else
    {
      // Prune this subtree.
      // First, make this node a leaf node.
      subtree_leaves_ = 1;
      subtree_leaves_error_ = error_;
      subtree_leaves_v_t_inv_ = v_t_inv_;

      delete left_;
      left_ = NULL;
      delete right_;
      right_ = NULL;

      // Pass information upward.
      return std::numeric_limits<double>::max();
    }
  }
}

// Check whether a given point is within the bounding box of this node (check
// generally done at the root, so its the bounding box of the data).
//
// Future improvement: Open up the range with epsilons on both sides where
// epsilon depends on the density near the boundary.
template<typename eT, typename cT>
inline bool DTree<eT, cT>::WithinRange(const arma::vec& query) const
{
  for (size_t i = 0; i < query.n_elem; ++i)
    if ((query[i] < minVals[i]) || (query[i] > maxVals[i]))
      return false;

  return true;
}


template<typename eT, typename cT>
cT DTree<eT, cT>::ComputeValue(VecType* query)
{
  assert(query->n_elem == maxVals.n_elem);

  if (root_ == 1) // If we are the root...
    // Check if the query is within range.
    if (!WithinRange(*query))
      return 0.0;

  if (subtree_leaves_ == 1)  // If we are a leaf...
    return ratio_ * v_t_inv_;
  else
  {
    if ((*query)[split_dim_] <= split_value_)
      // If left subtree, go to left child.
      return left_->ComputeValue(query);
    else  // If right subtree, go to right child
      return right_->ComputeValue(query);
  }
}


template<typename eT, typename cT>
void DTree<eT, cT>::WriteTree(size_t level, FILE *fp)
{
  if (subtree_leaves_ > 1)
  {
    fprintf(fp, "\n");
    for (size_t i = 0; i < level; ++i)
      fprintf(fp, "|\t");
    fprintf(fp, "Var. %zu > %lg", split_dim_, split_value_);

    right_->WriteTree(level + 1, fp);

    fprintf(fp, "\n");
    for (size_t i = 0; i < level; ++i)
      fprintf(fp, "|\t");
    fprintf(fp, "Var. %zu <= %lg ", split_dim_, split_value_);

    left_->WriteTree(level + 1, fp);
  }
  else // If we are a leaf...
  {
    fprintf(fp, ": f(x)=%Lg", (cT) ratio_ * v_t_inv_);
    if (bucket_tag_ != -1)
      fprintf(fp, " BT:%d", bucket_tag_);
  }
}


// Index the buckets for possible usage later.
template<typename eT, typename cT>
int DTree<eT, cT>::TagTree(int tag)
{
  if (subtree_leaves_ == 1)
  {
    bucket_tag_ = tag;
    return (tag + 1);
  }
  else
  {
    return right_->TagTree(left_->TagTree(tag));
  }
} // TagTree


template<typename eT, typename cT>
int DTree<eT, cT>::FindBucket(VecType* query)
{
  assert(query->n_elem == maxVals.n_elem);

  if (subtree_leaves_ == 1) // If we are a leaf...
  {
    return bucket_tag_;
  }
  else if ((*query)[split_dim_] <= split_value_)
  {
    // If left subtree, go to left child.
    return left_->FindBucket(query);
  }
  else // If right subtree, go to right child.
  {
    return right_->FindBucket(query);
  }
}


template<typename eT, typename cT>
void DTree<eT, cT>::ComputeVariableImportance(arma::Col<double> *imps)
{
  if (subtree_leaves_ == 1)
  {
    // If we are a leaf, do nothing.
    return;
  }
  else
  {
    // Compute the improvement in error because of the split.
    double error_improv = (double)
        (error_ - (left_->error() + right_->error()));
    (*imps)[split_dim_] += error_improv;
    left_->ComputeVariableImportance(imps);
    right_->ComputeVariableImportance(imps);
    return;
  }
}

}; // namespace det
}; // namespace mlpack

#endif // __MLPACK_METHODS_DET_DTREE_IMPL_HPP
