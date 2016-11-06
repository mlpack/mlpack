/**
 * @file dtree.cpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Ivan Georgiev (ivan@jonan.info) (sparsification and optimizations)
 *
 * Implementations of some declared functions in
 * the Density Estimation Tree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "dtree.hpp"
#include <stack>
#include <vector>

using namespace mlpack;
using namespace det;

namespace details
{
  /**
   * This one sorts and scand the given per-dimension extract and puts all splits
   * in a vector, that can easily be iterated afterwards. General implementation.
   */
  template <typename ElemType, typename MatType>
  void ExtractSplits(std::vector<std::pair<ElemType, size_t>>& splitVec,
                     const MatType& data,
                     size_t dim,
                     const size_t start,
                     const size_t end,
                     const size_t minLeafSize)
  {
    static_assert(
      std::is_same<typename MatType::elem_type, ElemType>::value == true,
      "The ElemType does not correspond to the matrix's element type."
                  );
    
    typedef std::pair<ElemType, size_t> SplitItem;
    const typename MatType::row_type dimVec =
      arma::sort(data(dim, arma::span(start, end - 1)));
    
    // Ensure the minimum leaf size on both sides. We need to figure out why
    // there are spikes if this minLeafSize is enforced here...
    for (size_t i = minLeafSize - 1; i < dimVec.n_elem - minLeafSize; ++i)
    {
      // This makes sense for real continuous data. This kinda corrupts the
      // data and estimation if the data is ordinal. Potentially we can fix
      // that by taking into account ordinality later in the min/max update,
      // but then we can end-up with a zero-volumed dimension. No good.
      const ElemType split = (dimVec[i] + dimVec[i + 1]) / 2.0;
      
      // Check if we can split here (two points are different)
      if (split != dimVec[i])
        splitVec.push_back(SplitItem(split, i + 1));
    }
  }

  // Now the custom arma::Mat implementation
  template <typename ElemType>
  void ExtractSplits(std::vector<std::pair<ElemType, size_t>>& splitVec,
                     const arma::Mat<ElemType>& data,
                     size_t dim,
                     const size_t start,
                     const size_t end,
                     const size_t minLeafSize)
  {
    typedef std::pair<ElemType, size_t> SplitItem;
    arma::vec dimVec = data(dim, arma::span(start, end - 1));
    
    // We sort these, in-place (it's a copy of the data, anyways).
    std::sort(dimVec.begin(), dimVec.end());
    
    for (size_t i = minLeafSize - 1; i < dimVec.n_elem - minLeafSize; ++i)
    {
      // This makes sense for real continuous data. This kinda corrupts the
      // data and estimation if the data is ordinal. Potentially we can fix
      // that by taking into account ordinality later in the min/max update,
      // but then we can end-up with a zero-volumed dimension. No good.
      const ElemType split = (dimVec[i] + dimVec[i + 1]) / 2.0;
      
      if (split != dimVec[i])
        splitVec.push_back(SplitItem(split, i + 1));
    }
  }
  
  // This the custom, sparse optimized implementation of the same routine.
  template <typename ElemType>
  void ExtractSplits(std::vector<std::pair<ElemType, size_t>>& splitVec,
                     const arma::SpMat<ElemType>& data,
                     size_t dim,
                     const size_t start,
                     const size_t end,
                     const size_t minLeafSize)
  {
    // It's common sense, but we also use it in a check later.
    Log::Assert(minLeafSize > 0);
    
    typedef std::pair<ElemType, size_t> SplitItem;
    const size_t n_elem = end - start;
    
    // Construct a vector of values.
    const arma::SpRow<ElemType> row = data(dim, arma::span(start, end - 1));
    std::vector<ElemType> valsVec(row.begin(), row.end());
    
    // ... and sort it!
    std::sort(valsVec.begin(), valsVec.end());

    // Now iterate over the values, taking account for the over-the-zeroes
    // jump and construct the splits vector.
    const size_t zeroes = n_elem - valsVec.size();
    ElemType lastVal = -std::numeric_limits<ElemType>::max();
    size_t padding = 0;

    for (size_t i = 0; i < valsVec.size(); ++i)
    {
      const ElemType newVal = valsVec[i];
      if (lastVal < ElemType(0) && newVal > ElemType(0) && zeroes > 0)
      {
        Log::Assert(padding == 0); // we should arrive here once!

        // the minLeafSize > 0 also guarantees we're not entering right at the start.
        if (i >= minLeafSize && i <= n_elem - minLeafSize)
          splitVec.push_back(SplitItem(lastVal / 2.0, i));

        padding = zeroes;
        lastVal = ElemType(0);
      }
      
      // the normal case
      if (i + padding >= minLeafSize && i + padding <= n_elem - minLeafSize)
      {
        // This makes sense for real continuous data.  This kinda corrupts the
        // data and estimation if the data is ordinal. Potentially we can fix
        // that by taking into account ordinality later in the min/max update,
        // but then we can end-up with a zero-volumed dimension. No good.
        const ElemType split = (lastVal + newVal) / 2.0;
        
        // Check if we can split here (two points are different)
        if (split != newVal)
          splitVec.push_back(SplitItem(split, i + padding));
      }
      
      lastVal = newVal;
    }
  }
};

template <typename MatType, typename TagType>
DTree<MatType, TagType>::DTree() :
    start(0),
    end(0),
    splitDim(size_t(-1)),
    splitValue(std::numeric_limits<ElemType>::max()),
    logNegError(-DBL_MAX),
    subtreeLeavesLogNegError(-DBL_MAX),
    subtreeLeaves(0),
    root(true),
    ratio(1.0),
    logVolume(-DBL_MAX),
    bucketTag(-1),
    alphaUpper(0.0),
    left(NULL),
    right(NULL)
{ /* Nothing to do. */ }


// Root node initializers

template <typename MatType, typename TagType>
DTree<MatType, TagType>::DTree(const StatType& maxVals,
                               const StatType& minVals,
                               const size_t totalPoints) :
    start(0),
    end(totalPoints),
    maxVals(maxVals),
    minVals(minVals),
    splitDim(size_t(-1)),
    splitValue(std::numeric_limits<ElemType>::max()),
    logNegError(LogNegativeError(totalPoints)),
    subtreeLeavesLogNegError(-DBL_MAX),
    subtreeLeaves(0),
    root(true),
    ratio(1.0),
    logVolume(-DBL_MAX),
    bucketTag(-1),
    alphaUpper(0.0),
    left(NULL),
    right(NULL)
{ /* Nothing to do. */ }

template <typename MatType, typename TagType>
DTree<MatType, TagType>::DTree(MatType & data) :
    start(0),
    end(data.n_cols),
    maxVals(arma::max(data, 1)),
    minVals(arma::min(data, 1)),
    splitDim(size_t(-1)),
    splitValue(std::numeric_limits<ElemType>::max()),
    subtreeLeavesLogNegError(-DBL_MAX),
    subtreeLeaves(0),
    root(true),
    ratio(1.0),
    logVolume(-DBL_MAX),
    bucketTag(-1),
    alphaUpper(0.0),
    left(NULL),
    right(NULL)
{
  logNegError = LogNegativeError(data.n_cols);
}

// Non-root node initializers
template <typename MatType, typename TagType>
DTree<MatType, TagType>::DTree(const StatType& maxVals,
                               const StatType& minVals,
                               const size_t start,
                               const size_t end,
                               const double logNegError) :
    start(start),
    end(end),
    maxVals(maxVals),
    minVals(minVals),
    splitDim(size_t(-1)),
    splitValue(std::numeric_limits<ElemType>::max()),
    logNegError(logNegError),
    subtreeLeavesLogNegError(-DBL_MAX),
    subtreeLeaves(0),
    root(false),
    ratio(1.0),
    logVolume(-DBL_MAX),
    bucketTag(-1),
    alphaUpper(0.0),
    left(NULL),
    right(NULL)
{ /* Nothing to do. */ }

template <typename MatType, typename TagType>
DTree<MatType, TagType>::DTree(const StatType& maxVals,
                               const StatType& minVals,
                               const size_t totalPoints,
                               const size_t start,
                               const size_t end) :
    start(start),
    end(end),
    maxVals(maxVals),
    minVals(minVals),
    splitDim(size_t(-1)),
    splitValue(std::numeric_limits<ElemType>::max()),
    logNegError(LogNegativeError(totalPoints)),
    subtreeLeavesLogNegError(-DBL_MAX),
    subtreeLeaves(0),
    root(false),
    ratio(1.0),
    logVolume(-DBL_MAX),
    bucketTag(-1),
    alphaUpper(0.0),
    left(NULL),
    right(NULL)
{ /* Nothing to do. */ }

template <typename MatType, typename TagType>
DTree<MatType, TagType>::~DTree()
{
  delete left;
  delete right;
}

// This function computes the log-l2-negative-error of a given node from the
// formula R(t) = log(|t|^2 / (N^2 V_t)).
template <typename MatType, typename TagType>
double DTree<MatType, TagType>::LogNegativeError(const size_t totalPoints) const
{
  // log(-|t|^2 / (N^2 V_t)) = log(-1) + 2 log(|t|) - 2 log(N) - log(V_t).
  double err = 2 * std::log((double) (end - start)) -
               2 * std::log((double) totalPoints);

  StatType valDiffs = maxVals - minVals;
  for (size_t i = 0;i < valDiffs.n_elem; ++i)
  {
    // Ignore very small dimensions to prevent overflow.
    if (valDiffs[i] > 1e-50)
      err -= std::log(valDiffs[i]);
  }

  return err;
}

// This function finds the best split with respect to the L2-error, by trying
// all possible splits.  The dataset is the full data set but the start and
// end are used to obtain the point in this node.
template <typename MatType, typename TagType>
bool DTree<MatType, TagType>::FindSplit(const MatType& data,
                                        size_t& splitDim,
                                        ElemType& splitValue,
                                        double& leftError,
                                        double& rightError,
                                        const size_t minLeafSize) const
{
  typedef std::pair<ElemType, size_t>   SplitItem;
  
  // Ensure the dimensionality of the data is the same as the dimensionality of
  // the bounding rectangle.
  Log::Assert(data.n_rows == maxVals.n_elem);
  Log::Assert(data.n_rows == minVals.n_elem);

  const size_t points = end - start;

  double minError = logNegError;
  bool splitFound = false;

  // Loop through each dimension.
#ifdef _WIN32
  #pragma omp parallel for default(shared)
  for (intmax_t dim = 0; dim < (intmax_t) maxVals.n_elem; ++dim)
#else
  #pragma omp parallel for default(shared)
  for (size_t dim = 0; dim < maxVals.n_elem; ++dim)
#endif
  {
    const ElemType min = minVals[dim];
    const ElemType max = maxVals[dim];

    // If there is nothing to split in this dimension, move on.
    if (max - min == 0.0)
      continue; // Skip to next dimension.

    // Find the log volume of all the other dimensions.
    const double volumeWithoutDim = logVolume - std::log(max - min);
    
    // Initializing all other stuff for this dimension.
    bool dimSplitFound = false;
    // Take an error estimate for this dimension.
    double minDimError = std::pow(points, 2.0) / (max - min);
    double dimLeftError = 0.0; // For -Wuninitialized.  These variables will
    double dimRightError = 0.0; // always be set to something else before use.
    ElemType dimSplitValue = 0.0;

    // Get the values for splitting. The old implementation:
    //   dimVec = data.row(dim).subvec(start, end - 1);
    //   dimVec = arma::sort(dimVec);
    // could be quite inefficient for sparse matrices, due to copy operations (3).
    // This one has custom implementation for dense and sparse matrices.

    std::vector<SplitItem> splitVec;
    details::ExtractSplits<ElemType>(splitVec, data, dim, start, end, minLeafSize);
    
    // Iterate on all the splits for this dimension
    for (typename std::vector<SplitItem>::iterator i = splitVec.begin();
         i != splitVec.end();
         ++i)
    {
      const ElemType split = i->first;
      const size_t position = i->second;
      
      // Another way of picking split is using this:
      //   split = leftsplit;
      if ((split - min > 0.0) && (max - split > 0.0))
      {
        // Ensure that the right node will have at least the minimum number of
        // points.
        Log::Assert((points - position) >= minLeafSize);

        // Now we have to see if the error will be reduced.  Simple manipulation
        // of the error function gives us the condition we must satisfy:
        //   |t_l|^2 / V_l + |t_r|^2 / V_r  >= |t|^2 / (V_l + V_r)
        // and because the volume is only dependent on the dimension we are
        // splitting, we can assume V_l is just the range of the left and V_r is
        // just the range of the right.
        double negLeftError = std::pow(position, 2.0) / (split - min);
        double negRightError = std::pow(points - position, 2.0) / (max - split);

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

    const double actualMinDimError = std::log(minDimError)
      - 2 * std::log((double) data.n_cols)
      - volumeWithoutDim;

#pragma omp critical (DTreeFindUpdate)
    if ((actualMinDimError > minError) && dimSplitFound)
    {
      // Calculate actual error (in logspace) by adding terms back to our
      // estimate.
      minError = actualMinDimError;
      splitDim = dim;
      splitValue = dimSplitValue;
      leftError = std::log(dimLeftError) - 2 * std::log((double) data.n_cols)
        - volumeWithoutDim;
      rightError = std::log(dimRightError) - 2 * std::log((double) data.n_cols)
        - volumeWithoutDim;
      splitFound = true;
    } // end if better split found in this dimension.
  }

  return splitFound;
}

template <typename MatType, typename TagType>
size_t DTree<MatType, TagType>::SplitData(MatType& data,
                                          const size_t splitDim,
                                          const ElemType splitValue,
                                          arma::Col<size_t>& oldFromNew) const
{
  // Swap all columns such that any columns with value in dimension splitDim
  // less than or equal to splitValue are on the left side, and all others are
  // on the right side.  A similar sort to this is also performed in
  // BinarySpaceTree construction (its comments are more detailed).
  size_t left = start;
  size_t right = end - 1;
  for (;;)
  {
    while (data(splitDim, left) <= splitValue)
      ++left;
    while (data(splitDim, right) > splitValue)
      --right;

    if (left > right)
      break;

    data.swap_cols(left, right);

    // Store the mapping from old to new. Do not put std::swap here...
    const size_t tmp = oldFromNew[left];
    oldFromNew[left] = oldFromNew[right];
    oldFromNew[right] = tmp;
  }

  // This now refers to the first index of the "right" side.
  return left;
}

// Greedily expand the tree
template <typename MatType, typename TagType>
double DTree<MatType, TagType>::Grow(MatType& data,
                                     arma::Col<size_t>& oldFromNew,
                                     const bool useVolReg,
                                     const size_t maxLeafSize,
                                     const size_t minLeafSize)
{
  Log::Assert(data.n_rows == maxVals.n_elem);
  Log::Assert(data.n_rows == minVals.n_elem);

  double leftG, rightG;

  // Compute points ratio.
  ratio = (double) (end - start) / (double) oldFromNew.n_elem;

  // Compute the log of the volume of the node.
  logVolume = 0;
  for (size_t i = 0; i < maxVals.n_elem; ++i)
    if (maxVals[i] - minVals[i] > 0.0)
      logVolume += std::log(maxVals[i] - minVals[i]);

  // Check if node is large enough to split.
  if ((size_t) (end - start) > maxLeafSize)
  {
    // Find the split.
    size_t dim;
    double splitValueTmp;
    double leftError, rightError;
    if (FindSplit(data, dim, splitValueTmp, leftError, rightError, minLeafSize))
    {
      // Move the data around for the children to have points in a node lie
      // contiguously (to increase efficiency during the training).
      const size_t splitIndex = SplitData(data, dim, splitValueTmp, oldFromNew);

      // Make max and min vals for the children.
      StatType maxValsL(maxVals);
      StatType maxValsR(maxVals);
      StatType minValsL(minVals);
      StatType minValsR(minVals);

      maxValsL[dim] = splitValueTmp;
      minValsR[dim] = splitValueTmp;

      // Store split dim and split val in the node.
      splitValue = splitValueTmp;
      splitDim = dim;

      // Recursively grow the children.
      left = new DTree(maxValsL, minValsL, start, splitIndex, leftError);
      right = new DTree(maxValsR, minValsR, splitIndex, end, rightError);

      leftG = left->Grow(data, oldFromNew, useVolReg, maxLeafSize,
                         minLeafSize);
      rightG = right->Grow(data, oldFromNew, useVolReg, maxLeafSize,
                           minLeafSize);

      // Store values of R(T~) and |T~|.
      subtreeLeaves = left->SubtreeLeaves() + right->SubtreeLeaves();

      // Find the log negative error of the subtree leaves.  This is kind of an
      // odd one because we don't want to represent the error in non-log-space,
      // but we have to calculate log(E_l + E_r).  So we multiply E_l and E_r by
      // V_t (remember E_l has an inverse relationship to the volume of the
      // nodes) and then subtract log(V_t) at the end of the whole expression.
      // As a result we do leave log-space, but the largest quantity we
      // represent is on the order of (V_t / V_i) where V_i is the smallest leaf
      // node below this node, which depends heavily on the depth of the tree.
      subtreeLeavesLogNegError = std::log(
          std::exp(logVolume + left->SubtreeLeavesLogNegError()) +
          std::exp(logVolume + right->SubtreeLeavesLogNegError()))
          - logVolume;
    }
    else
    {
      // No split found so make a leaf out of it.
      subtreeLeaves = 1;
      subtreeLeavesLogNegError = logNegError;
    }
  }
  else
  {
    // We can make this a leaf node.
    Log::Assert((size_t) (end - start) >= minLeafSize);
    subtreeLeaves = 1;
    subtreeLeavesLogNegError = logNegError;
  }

  // If this is a leaf, do not compute g_k(t); otherwise compute, store, and
  // propagate min(g_k(t_L), g_k(t_R), g_k(t)), unless t_L and/or t_R are
  // leaves.
  if (subtreeLeaves == 1)
  {
    return std::numeric_limits<double>::max();
  }
  else
  {
    const double range = maxVals[splitDim] - minVals[splitDim];
    const double leftRatio = (splitValue - minVals[splitDim]) / range;
    const double rightRatio = (maxVals[splitDim] - splitValue) / range;

    const size_t leftPow = std::pow((double) (left->End() - left->Start()), 2);
    const size_t rightPow = std::pow((double) (right->End() - right->Start()),
        2);
    const size_t thisPow = std::pow((double) (end - start), 2);

    double tmpAlphaSum = leftPow / leftRatio + rightPow / rightRatio - thisPow;

    if (left->SubtreeLeaves() > 1)
    {
      const double exponent = 2 * std::log((double) data.n_cols) + logVolume +
          left->AlphaUpper();

      // Whether or not this will overflow is highly dependent on the depth of
      // the tree.
      tmpAlphaSum += std::exp(exponent);
    }

    if (right->SubtreeLeaves() > 1)
    {
      const double exponent = 2 * std::log((double) data.n_cols)
        + logVolume
        + right->AlphaUpper();

      tmpAlphaSum += std::exp(exponent);
    }

    alphaUpper = std::log(tmpAlphaSum) - 2 * std::log((double) data.n_cols)
      - logVolume;

    double gT;
    if (useVolReg)
    {
      // This is wrong for now!
      gT = alphaUpper;// / (subtreeLeavesVTInv - vTInv);
    }
    else
    {
      gT = alphaUpper - std::log((double) (subtreeLeaves - 1));
    }

    return std::min(gT, std::min(leftG, rightG));
  }

  // We need to compute (c_t^2) * r_t for all subtree leaves; this is equal to
  // n_t ^ 2 / r_t * n ^ 2 = -error.  Therefore the value we need is actually
  // -1.0 * subtreeLeavesError.
}


template <typename MatType, typename TagType>
double DTree<MatType, TagType>::PruneAndUpdate(const double oldAlpha,
                                               const size_t points,
                                               const bool useVolReg)
{
  // Compute gT.
  if (subtreeLeaves == 1) // If we are a leaf...
  {
    return std::numeric_limits<double>::max();
  }
  else
  {
    // Compute gT value for node t.
    volatile double gT;
    if (useVolReg)
      gT = alphaUpper;// - std::log(subtreeLeavesVTInv - vTInv);
    else
      gT = alphaUpper - std::log((double) (subtreeLeaves - 1));

    if (gT > oldAlpha)
    {
      // Go down the tree and update accordingly.  Traverse the children.
      double leftG = left->PruneAndUpdate(oldAlpha, points, useVolReg);
      double rightG = right->PruneAndUpdate(oldAlpha, points, useVolReg);

      // Update values.
      subtreeLeaves = left->SubtreeLeaves() + right->SubtreeLeaves();

      // Find the log negative error of the subtree leaves.  This is kind of an
      // odd one because we don't want to represent the error in non-log-space,
      // but we have to calculate log(E_l + E_r).  So we multiply E_l and E_r by
      // V_t (remember E_l has an inverse relationship to the volume of the
      // nodes) and then subtract log(V_t) at the end of the whole expression.
      // As a result we do leave log-space, but the largest quantity we
      // represent is on the order of (V_t / V_i) where V_i is the smallest leaf
      // node below this node, which depends heavily on the depth of the tree.
      subtreeLeavesLogNegError = std::log(
          std::exp(logVolume + left->SubtreeLeavesLogNegError()) +
          std::exp(logVolume + right->SubtreeLeavesLogNegError()))
          - logVolume;

      // Recalculate upper alpha.
      const double range = maxVals[splitDim] - minVals[splitDim];
      const double leftRatio = (splitValue - minVals[splitDim]) / range;
      const double rightRatio = (maxVals[splitDim] - splitValue) / range;

      const size_t leftPow = std::pow((double) (left->End() - left->Start()),
          2);
      const size_t rightPow = std::pow((double) (right->End() - right->Start()),
          2);
      const size_t thisPow = std::pow((double) (end - start), 2);

      double tmpAlphaSum = leftPow / leftRatio + rightPow / rightRatio -
          thisPow;

      if (left->SubtreeLeaves() > 1)
      {
        const double exponent = 2 * std::log((double) points) + logVolume +
            left->AlphaUpper();

        // Whether or not this will overflow is highly dependent on the depth of
        // the tree.
        tmpAlphaSum += std::exp(exponent);
      }

      if (right->SubtreeLeaves() > 1)
      {
        const double exponent = 2 * std::log((double) points) + logVolume +
            right->AlphaUpper();

        tmpAlphaSum += std::exp(exponent);
      }

      alphaUpper = std::log(tmpAlphaSum) - 2 * std::log((double) points) -
          logVolume;

      // Update gT value.
      if (useVolReg)
      {
        // This is incorrect.
        gT = alphaUpper; // / (subtreeLeavesVTInv - vTInv);
      }
      else
      {
        gT = alphaUpper - std::log((double) (subtreeLeaves - 1));
      }

      Log::Assert(gT < std::numeric_limits<double>::max());

      return std::min((double) gT, std::min(leftG, rightG));
    }
    else
    {
      // Prune this subtree.
      // First, make this node a leaf node.
      subtreeLeaves = 1;
      subtreeLeavesLogNegError = logNegError;

      delete left;
      delete right;

      left = NULL;
      right = NULL;

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
template <typename MatType, typename TagType>
bool DTree<MatType, TagType>::WithinRange(const VecType& query) const
{
  for (size_t i = 0; i < query.n_elem; ++i)
    if ((query[i] < minVals[i]) || (query[i] > maxVals[i]))
      return false;

  return true;
}


template <typename MatType, typename TagType>
double DTree<MatType, TagType>::ComputeValue(const VecType& query) const
{
  Log::Assert(query.n_elem == maxVals.n_elem);

  if (root == 1) // If we are the root...
  {
    // Check if the query is within range.
    if (!WithinRange(query))
      return 0.0;
  }

  if (subtreeLeaves == 1)  // If we are a leaf...
  {
    return std::exp(std::log(ratio) - logVolume);
  }
  else
  {
    // Return either of the two children - left or right, depending on the splitValue
    return (query[splitDim] <= splitValue) ?
      left->ComputeValue(query) :
      right->ComputeValue(query);
  }

  return 0.0;
}


// Index the buckets for possible usage later.
template <typename MatType, typename TagType>
TagType DTree<MatType, TagType>::TagTree(const TagType& tag)
{
  if (subtreeLeaves == 1)
  {
    // Only label leaves.
    bucketTag = tag;
    return (tag + 1);
  }
  else
  {
    return right->TagTree(left->TagTree(tag));
  }
}


template <typename MatType, typename TagType>
TagType DTree<MatType, TagType>::FindBucket(const VecType& query) const
{
  Log::Assert(query.n_elem == maxVals.n_elem);

  if (subtreeLeaves == 1) // If we are a leaf...
  {
    return bucketTag;
  }
  else
  {
    // Return the tag from either of the two children - left or right.
    return (query[splitDim] <= splitValue) ?
      left->FindBucket(query) :
      right->FindBucket(query);
  }
}

template <typename MatType, typename TagType>
void
DTree<MatType, TagType>::ComputeVariableImportance(arma::vec& importances) const
{
  // Clear and set to right size.
  importances.zeros(maxVals.n_elem);

  std::stack<const DTree*> nodes;
  nodes.push(this);

  while(!nodes.empty())
  {
    const DTree& curNode = *nodes.top();
    nodes.pop();

    if (curNode.subtreeLeaves == 1)
      continue; // Do nothing for leaves.

    // The way to do this entirely in log-space is (at this time) somewhat
    // unclear.  So this risks overflow.
    importances[curNode.SplitDim()] += (-std::exp(curNode.LogNegError()) -
        (-std::exp(curNode.Left()->LogNegError()) +
         -std::exp(curNode.Right()->LogNegError())));

    nodes.push(curNode.Left());
    nodes.push(curNode.Right());
  }
}

template <typename MatType, typename TagType>
template <typename Archive>
void DTree<MatType, TagType>::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;
  
  ar & CreateNVP(start, "start");
  ar & CreateNVP(end, "end");
  ar & CreateNVP(maxVals, "maxVals");
  ar & CreateNVP(minVals, "minVals");
  ar & CreateNVP(splitDim, "splitDim");
  ar & CreateNVP(splitValue, "splitValue");
  ar & CreateNVP(logNegError, "logNegError");
  ar & CreateNVP(subtreeLeavesLogNegError, "subtreeLeavesLogNegError");
  ar & CreateNVP(subtreeLeaves, "subtreeLeaves");
  ar & CreateNVP(root, "root");
  ar & CreateNVP(ratio, "ratio");
  ar & CreateNVP(logVolume, "logVolume");
  ar & CreateNVP(bucketTag, "bucketTag");
  ar & CreateNVP(alphaUpper, "alphaUpper");
  
  if (Archive::is_loading::value)
  {
    if (left)
      delete left;
    if (right)
      delete right;
  }
  
  ar & CreateNVP(left, "left");
  ar & CreateNVP(right, "right");
}

