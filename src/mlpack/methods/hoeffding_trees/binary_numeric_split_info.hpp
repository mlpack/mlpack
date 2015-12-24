/**
 * @file binary_numeric_split_info.hpp
 * @author Ryan Curtin
 *
 * After a binary numeric split has been made, this holds information on the
 * split (just the split point).
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
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_BINARY_NUMERIC_SPLIT_INFO_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_BINARY_NUMERIC_SPLIT_INFO_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

template<typename ObservationType = double>
class BinaryNumericSplitInfo
{
 public:
  BinaryNumericSplitInfo() { /* Nothing to do. */ }
  BinaryNumericSplitInfo(const ObservationType& splitPoint) :
      splitPoint(splitPoint) { /* Nothing to do. */ }

  template<typename eT>
  size_t CalculateDirection(const eT& value) const
  {
    return (value < splitPoint) ? 0 : 1;
  }

  //! Serialize the split (save/load the split points).
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(splitPoint, "splitPoint");
  }

 private:
  ObservationType splitPoint;
};

} // namespace tree
} // namespace mlpack

#endif
