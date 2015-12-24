/**
 * @file categorical_split_info.hpp
 * @author Ryan Curtin
 *
 * After a categorical split has been made, this holds information on the split.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_CATEGORICAL_SPLIT_INFO_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_CATEGORICAL_SPLIT_INFO_HPP

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
