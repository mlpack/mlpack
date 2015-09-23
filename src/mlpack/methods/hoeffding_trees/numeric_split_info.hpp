/**
 * @file numeric_split_info.hpp
 * @author Ryan Curtin
 *
 * After a numeric split has been made, this holds information on the split.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_NUMERIC_SPLIT_INFO_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_NUMERIC_SPLIT_INFO_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

// This doesn't do anything yet.
class NumericSplitInfo
{
 public:
  NumericSplitInfo() { }

  template<typename eT>
  static size_t CalculateDirection(const eT& /* value */) { return 0; }
};

} // namespace tree
} // namespace mlpack

#endif
