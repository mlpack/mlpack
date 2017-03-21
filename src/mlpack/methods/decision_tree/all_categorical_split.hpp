/**
 * @file all_categorical_split.hpp
 * @author Ryan Curtin
 *
 * This file defines a tree splitter that split a categorical feature into all
 * of the possible categories.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_ALL_CATEGORICAL_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_ALL_CATEGORICAL_SPLIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace tree {

/**
 * The AllCategoricalSplit is a splitting function that will split categorical
 * features into many children: one child for each category.
 *
 * @tparam FitnessFunction Fitness function to evaluate gain with.
 */
template<typename FitnessFunction>
class AllCategoricalSplit
{
 public:
  // No extra info needed for split.
  template<typename ElemType>
  class AuxiliarySplitInfo { };

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on 'bestGain', then we return the improved gain.  Otherwise we
   * return the value 'bestGain'.  If a split is made, then classProbabilities
   * and aux may be modified.  For this particular split type, aux will be empty
   * and classProbabilities will hold one element---the number of children.
   *
   * @param bestGain Best gain seen so far (we'll only split if we find gain
   *      better than this).
   * @param data The dimension of data points to check for a split in.
   * @param numCategories Number of categories in the categorical data.
   * @param labels Labels for each point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in a leaf node for
   *      splitting.
   * @param classProbabilities Class probabilities vector, which may be filled
   *      with split information a successful split.
   * @param aux Auxiliary split information, which may be modified on a
   *      successful split.
   */
  template<typename VecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const size_t numCategories,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const size_t minimumLeafSize,
      arma::Col<typename VecType::elem_type>& classProbabilities,
      AuxiliarySplitInfo<typename VecType::elem_type>& aux);

  /**
   * Return the number of children in the split.
   *
   * @param classProbabilities Auxiliary information for the split.
   * @param aux (Unused) auxiliary information for the split.
   */
  template<typename ElemType>
  static size_t NumChildren(const arma::Col<ElemType>& classProbabilities,
                            const AuxiliarySplitInfo<ElemType>& /* aux */);

  /**
   * Calculate the direction a point should percolate to.
   *
   * @param classProbabilities Auxiliary information for the split.
   * @param aux (Unused) auxiliary information for the split.
   */
  template<typename ElemType>
  static size_t CalculateDirection(
      const ElemType& point,
      const arma::Col<ElemType>& classProbabilities,
      const AuxiliarySplitInfo<ElemType>& /* aux */);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "all_categorical_split_impl.hpp"

#endif

