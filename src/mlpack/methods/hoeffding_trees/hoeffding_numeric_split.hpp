/**
 * @file hoeffding_numeric_split.hpp
 * @author Ryan Curtin
 *
 * A numeric feature split for Hoeffding trees.  At the moment it does nothing.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP

#include <mlpack/core.hpp>
#include "numeric_split_info.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
class HoeffdingNumericSplit
{
 public:
  typedef NumericSplitInfo SplitInfo;

  HoeffdingNumericSplit();

  template<typename eT>
  void Train(eT /* value */, const size_t /* label */) { }

  double EvaluateFitnessFunction() const { return 0.0; }

  // Does nothing for now.
  template<typename StreamingDecisionTreeType>
  void CreateChildren(std::vector<StreamingDecisionTreeType>& children,
                      const data::DatasetInfo& datasetInfo,
                      SplitInfo& splitInfo) { } // Nothing to do.

  size_t MajorityClass() const { return 0; } // Nothing yet.
};

} // namespace tree
} // namespace mlpack

#endif
