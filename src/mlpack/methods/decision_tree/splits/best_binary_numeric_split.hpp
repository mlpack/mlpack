/**
 * @file methods/decision_tree/splits/best_binary_numeric_split.hpp
 * @author Ryan Curtin
 *
 * A tree splitter that finds the best binary numeric split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_SPLITS_BEST_BINARY_NUMERIC_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_SPLITS_BEST_BINARY_NUMERIC_SPLIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/decision_tree/fitness_functions/mse_gain.hpp>

#include <mlpack/core/util/sfinae_utility.hpp>

namespace mlpack {

// This gives us a HasBinaryGains<T, U> type (where U is a function pointer)
// we can use with SFINAE to catch when a type has a BinaryGains(...) function.
HAS_MEM_FUNC(BinaryGains, HasBinaryGains);

// This struct will have `value` set to `true` if a BinaryGains() function of
// the right signature is detected.  We only check for BinaryGains(), and not
// BinaryScanInitialize() or BinaryStep(), because those two are template
// members functions and would make this check far more difficult.
//
// The unused UseWeights template parameter is necessary to ensure that the
// compiler thinks the result `value` depends on a parameter specific to the
// SplitIfBetter() function in BestBinaryNumericSplit().
template<typename T, bool /* UseWeights */>
struct HasOptimizedBinarySplitForms
{
  static const bool value = HasBinaryGains<T,
      std::tuple<double, double>(T::*)()>::value;
};

/**
 * The BestBinaryNumericSplit is a splitting function for decision trees that
 * will exhaustively search a numeric dimension for the best binary split.
 *
 * @tparam FitnessFunction Fitness function to use to calculate gain.
 */
template<typename FitnessFunction>
class BestBinaryNumericSplit
{
 public:
  // No extra info needed for split.
  class AuxiliarySplitInfo { };

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on 'bestGain', then we return the improved gain.  Otherwise we
   * return the value 'bestGain'.  If a split is made, then splitInfo and aux
   * may be modified.
   *
   * This overload is used only for classification tasks.
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
      AuxiliarySplitInfo& aux);

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on 'bestGain', then we return the improved gain.  Otherwise we
   * return the value 'bestGain'.  If a split is made, then splitInfo and aux
   * may be modified.
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
   * @param fitnessFunction The FitnessFunction object instance. It is used to
   *      evaluate the gain for the split.
   */
  template<bool UseWeights, typename VecType, typename ResponsesType,
           typename WeightVecType>
  static std::enable_if_t<
      !HasOptimizedBinarySplitForms<FitnessFunction, UseWeights>::value,
      double>
  SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const ResponsesType& responses,
      const WeightVecType& weights,
      const size_t minimumLeafSize,
      const double minimumGainSplit,
      arma::vec& splitInfo,
      AuxiliarySplitInfo& aux,
      FitnessFunction& fitnessFunction);

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on 'bestGain', then we return the improved gain.  Otherwise we
   * return the value 'bestGain'.  If a split is made, then splitInfo and aux
   * may be modified.
   *
   * This overload is specialized for any fitness function that implements
   * BinaryScanInitialize(), BinaryStep() and BinaryGains() functions.
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
   */
  template<bool UseWeights, typename VecType, typename ResponsesType,
          typename WeightVecType>
  static std::enable_if_t<
      HasOptimizedBinarySplitForms<FitnessFunction, UseWeights>::value,
      double>
  SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const ResponsesType& responses,
      const WeightVecType& weights,
      const size_t minimumLeafSize,
      const double minimumGainSplit,
      arma::vec& splitInfo,
      AuxiliarySplitInfo& /* aux */,
      FitnessFunction& fitnessFunction);

  /**
   * If a split was found, returns the number of children of the split. 
   * Otherwise returns zero. A binary split always has two children.
   */
  static size_t NumChildren(const arma::vec& splitInfo,
                            const AuxiliarySplitInfo& /* aux */)
  {
    return splitInfo.n_elem == 0 ? 0 : 2;
  }

  /**
   * In the case that a split was found, given a point, calculate 
   * which child it should go to (left or right). Otherwise if 
   * there was no split, returns SIZE_MAX.
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
#include "best_binary_numeric_split_impl.hpp"

#endif
