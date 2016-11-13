/**
 * @file categorical_split_info.hpp
 * @author Ryan Curtin
 *
 * After a categorical split has been made, this holds information on the split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_CATEGORICAL_SPLIT_INFO_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_CATEGORICAL_SPLIT_INFO_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

class CategoricalSplitInfo
{
 public:
  CategoricalSplitInfo(const size_t /* categories */) { }

  template<typename eT>
  static size_t CalculateDirection(const eT& value)
  {
    // We have a child for each categorical value, and value should be in the
    // range [0, categories).
    return size_t(value);
  }

  //! Serialize the object.  (Nothing needs to be saved.)
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace tree
} // namespace mlpack

#endif
