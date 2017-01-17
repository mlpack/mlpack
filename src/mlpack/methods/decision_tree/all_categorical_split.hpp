/**
 * @file all_categorical_split.hpp
 * @author Ryan Curtin
 *
 * This file defines a tree splitter that split a categorical feature into all
 * of the possible categories.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_ALL_CATEGORICAL_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_ALL_CATEGORICAL_SPLIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace tree {

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
   */
  template<typename ElemType>
  static size_t NumChildren(const arma::Col<ElemType>& classProbabilities,
                            const AuxiliarySplitInfo<ElemType>& /* aux */);

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

