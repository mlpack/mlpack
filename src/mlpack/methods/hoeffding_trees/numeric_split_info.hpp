/**
 * @file numeric_split_info.hpp
 * @author Ryan Curtin
 *
 * After a numeric split has been made, this holds information on the split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_NUMERIC_SPLIT_INFO_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_NUMERIC_SPLIT_INFO_HPP

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
