/**
 * @file categorical_split_info.hpp
 * @author Ryan Curtin
 *
 * After a categorical split has been made, this holds information on the split.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_CATEGORICAL_SPLIT_INFO_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_CATEGORICAL_SPLIT_INFO_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

class CategoricalSplitInfo
{
 public:
  CategoricalSplitInfo(const size_t categories) : categories(categories) { }

  template<typename eT>
  void CalculateDirection(const eT& value)
  {
    // We have a child for each categorical value, and value should be in the
    // range [0, categories).
    return value;
  }

 private:
  const size_t categories;
};

} // namespace tree
} // namespace mlpack

#endif
