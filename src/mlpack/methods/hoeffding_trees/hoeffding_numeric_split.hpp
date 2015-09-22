/**
 * @file hoeffding_numeric_split.hpp
 * @author Ryan Curtin
 *
 * A numeric feature split for Hoeffding trees.  At the moment it does nothing.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
class HoeffdingNumericSplit
{
 public:
  typedef size_t SplitInfo;

  HoeffdingNumericSplit();

  template<typename eT>
  void Train(eT /* value */, const size_t /* label */) { }

  double EvaluateFitnessFunction() const { return 0.0; }
};

} // namespace tree
} // namespace mlpack

#endif
