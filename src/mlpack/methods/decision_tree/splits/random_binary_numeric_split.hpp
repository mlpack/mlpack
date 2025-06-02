/**
 * @file methods/decision_tree/splits/random_binary_numeric_split.hpp
 * @author Rishabh Garg
 *
 * A tree splitter that finds a random binary numeric split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_SPLITS_RANDOM_BINARY_NUMERIC_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_SPLITS_RANDOM_BINARY_NUMERIC_SPLIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The RandomBinaryNumericSplit is a splitting function for decision trees that
 * will split based on a randomly selected point between the minimum
 * and maximum value of the numerical dimension.
 *
 * @tparam FitnessFunction Fitness function to use to calculate gain.
 */
template<typename FitnessFunction>
class RandomBinaryNumericSplit
{
 public:
  // No extra info needed for split.
  class AuxiliarySplitInfo { };

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on 'bestGain', then we return the improved gain.  Otherwise we
   * return the value 'bestGain'.  If a split is made, then splitInfo
   * and aux may be modified.
   *
   * This overload is used only for classification tasks.
   *
   * @code
   * @article{10.1007/s10994-006-6226-1,
   *   author = {Geurts, Pierre and Ernst, Damien and Wehenkel, Louis},
   *   title = {Extremely Randomized Trees},
   *   year = {2006},
   *   issue_date = {April 2006},
   *   publisher = {Kluwer Academic Publishers},
   *   address = {USA},
   *   volume = {63},
   *   number = {1},
   *   issn = {0885-6125},
   *   url = {https://doi.org/10.1007/s10994-006-6226-1},
   *   doi = {10.1007/s10994-006-6226-1},
   *   journal = {Mach. Learn.},
   *   month = apr,
   *   pages = {3–42},
   *   numpages = {40},
   * }
   * @endcode
   *
   * @param bestGain Best gain seen so far (we'll only split if we find gain
   *      better than this).
   * @param data The dimension of data points to check for a split in.
   * @param labels Labels for each point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Weights associated with labels.
   * @param minimumLeafSize Minimum number of points in a leaf node for
   *      splitting.
   * @param minimumGainSplit Minimum gain split.
   * @param splitInfo Stores split information on a successful split.
   * @param aux Auxiliary split information, which may be modified on a
   *      successful split.
   * @param splitIfBetterGain When set to true, it will split only when gain is
   *      better than the current best gain. Otherwise, it always makes a
   *      split regardless of gain.
   */
  template<bool UseWeights, typename VecType, typename WeightVecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const WeightVecType& weights,
      const size_t minimumLeafSize,
      const double minimumGainSplit,
      arma::vec& splitInfo,
      AuxiliarySplitInfo& aux,
      const bool splitIfBetterGain = false);

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on 'bestGain', then we return the improved gain.  Otherwise we
   * return the value 'bestGain'.  If a split is made, then splitInfo
   * and aux may be modified.
   *
   * This overload is used only for regression tasks.
   *
   * @param bestGain Best gain seen so far (we'll only split if we find gain
   *      better than this).
   * @param data The dimension of data points to check for a split in.
   * @param responses Responses for each point.
   * @param weights Weights associated with responses.
   * @param minimumLeafSize Minimum number of points in a leaf node for
   *      splitting.
   * @param minimumGainSplit Minimum gain split.
   * @param splitInfo Stores split information on a successful split.
   * @param aux Auxiliary split information, which may be modified on a
   *      successful split.
   * @param fitnessFunction The FitnessFunction object instance. It it used to
   *      evaluate the gain for the split.
   * @param splitIfBetterGain When set to true, it will split only when gain is
   *      better than the current best gain. Otherwise, it always makes a
   *      split regardless of gain.
   */
  template<bool UseWeights, typename VecType, typename WeightVecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const arma::rowvec& responses,
      const WeightVecType& weights,
      const size_t minimumLeafSize,
      const double minimumGainSplit,
      arma::vec& splitInfo,
      AuxiliarySplitInfo& aux,
      FitnessFunction& fitnessFunction,
      const bool splitIfBetterGain = false);

  /**
   * If a split was found, returns the number of children of the split.
   * Otherwise returns zero. A binary split always has two children.
   *
   * @param splitInfo Auxiliary information for the split.
   * @param aux Auxiliary split information, which may be modified on a
   *      successful split.
   */
  static size_t NumChildren(const arma::vec& splitInfo,
                            const AuxiliarySplitInfo& /* aux */)
  {
    return splitInfo.n_elem == 0 ? 0 : 2;
  }

  /**
   * If a split was found, given a point, calculate which child it should 
   * go to (left or right). Otherwise if there was no split, returns SIZE_MAX.
   *
   * @param point Point to calculate direction of.
   * @param splitInfo Auxiliary information for the split.
   * @param * (aux) Auxiliary information for the split (Unused).
   */
  template<typename ElemType>
  static size_t CalculateDirection(
      const ElemType& point,
      const arma::vec& splitInfo,
      const AuxiliarySplitInfo& /* aux */);
};

} // namespace mlpack

// Include implementation.
#include "random_binary_numeric_split_impl.hpp"

#endif
