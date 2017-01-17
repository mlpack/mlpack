/**
 * @file best_binary_numeric_split.hpp
 * @author Ryan Curtin
 *
 * A tree splitter that finds the best binary numeric split.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_BEST_BINARY_NUMERIC_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_BEST_BINARY_NUMERIC_SPLIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
class BestBinaryNumericSplit
{
 public:
  // No extra info needed for split.
  template<typename ElemType>
  class AuxiliarySplitInfo { };

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on 'bestGain', then we return the improved gain.  Otherwise we
   * return the value 'bestGain'.  If a split is made, then classProbabilities
   * and aux may be modified.
   */
  template<typename VecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const size_t minimumLeafSize,
      arma::Col<typename VecType::elem_type>& classProbabilities,
      AuxiliarySplitInfo<typename VecType::elem_type>& aux);

  /**
   * Returns 2, since the binary split always has two children.
   */
  template<typename ElemType>
  static size_t NumChildren(const arma::Col<ElemType>& /* classProbabilities */,
                            const AuxiliarySplitInfo<ElemType>& /* aux */)
  {
    return 2;
  }

  template<typename ElemType>
  static size_t CalculateDirection(
      const ElemType& point,
      const arma::Col<ElemType>& classProbabilities,
      const AuxiliarySplitInfo<ElemType>& /* aux */);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "best_binary_numeric_split_impl.hpp"

#endif
