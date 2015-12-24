/**
 * @file numeric_split_info.hpp
 * @author Ryan Curtin
 *
 * After a numeric split has been made, this holds information on the split.
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
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_NUMERIC_SPLIT_INFO_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_NUMERIC_SPLIT_INFO_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

template<typename ObservationType = double>
class NumericSplitInfo
{
 public:
  NumericSplitInfo() { /* Nothing to do. */ }
  NumericSplitInfo(const arma::Col<ObservationType>& splitPoints) :
      splitPoints(splitPoints) { /* Nothing to do. */ }

  template<typename eT>
  size_t CalculateDirection(const eT& value) const
  {
    // What bin does the point fall into?
    size_t bin = 0;
    while (bin < splitPoints.n_elem && value > splitPoints[bin])
      ++bin;

    return bin;
  }

  //! Serialize the split (save/load the split points).
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(splitPoints, "splitPoints");
  }

 private:
  arma::Col<ObservationType> splitPoints;
};

} // namespace tree
} // namespace mlpack

#endif
