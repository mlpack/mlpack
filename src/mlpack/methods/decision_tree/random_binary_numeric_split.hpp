/**
 * @file methods/decision_tree/random_binary_numeric_split.hpp
 * @author Rishabh Garg
 *
 * A tree splitter that finds a random binary numeric split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_RANDOM_BINARY_NUMERIC_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_RANDOM_BINARY_NUMERIC_SPLIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace tree {

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
   * return the value 'bestGain'.  If a split is made, then classProbabilities
   * and aux may be modified.
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
   * @param classProbabilities Class probabilities vector, which may be filled
   *      with split information a successful split.
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
      arma::vec& classProbabilities,
      AuxiliarySplitInfo& aux,
      const bool splitIfBetterGain = false);

  /**
   * Returns 2, since the binary split always has two children.
   *
   * @param classProbabilities Class probabilities vector, which may be filled
   *      with split information a successful split. (Not used here.)
   * @param aux Auxiliary split information, which may be modified on a
   *      successful split. (Not used here.)
   */
  static size_t NumChildren(const arma::vec& /* classProbabilities */,
                            const AuxiliarySplitInfo& /* aux */)
  {
    return 2;
  }

  /**
   * Given a point, calculate which child it should go to (left or right).
   *
   * @param point Point to calculate direction of.
   * @param classProbabilities Auxiliary information for the split.
   * @param * (aux) Auxiliary information for the split (Unused).
   */
  template<typename ElemType>
  static size_t CalculateDirection(
      const ElemType& point,
      const arma::vec& classProbabilities,
      const AuxiliarySplitInfo& /* aux */);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "random_binary_numeric_split_impl.hpp"

#endif
