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

template<typename ObservationType = double>
class NumericSplitInfo
{
 public:
  NumericSplitInfo() { /* Nothing to do. */ }
  NumericSplitInfo(arma::Col<ObservationType>& splitPoints) :
      splitPoints(splitPoints) { /* Nothing to do. */ }

  template<typename eT>
  size_t CalculateDirection(const eT& value) const
  {
    // What bin does the point fall into?
    size_t bin = 0;
    while (value > splitPoints[bin] && bin < splitPoints.n_elem)
      ++bin;

    return bin;
  }

 private:
  arma::Col<ObservationType> splitPoints;
};

} // namespace tree
} // namespace mlpack

#endif
