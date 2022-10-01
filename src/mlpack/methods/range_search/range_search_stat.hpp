/**
 * @file methods/range_search/range_search_stat.hpp
 * @author Ryan Curtin
 *
 * Statistic class for RangeSearch, which just holds the last visited node and
 * the corresponding base case result.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_STAT_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_STAT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Statistic class for RangeSearch, to be set to the StatisticType of the tree
 * type that range search is being performed with.  This class just holds the
 * last visited node and the corresponding base case result.
 */
class RangeSearchStat
{
 public:
  /**
   * Initialize the statistic.
   */
  RangeSearchStat() : lastDistance(0.0) { }

  /**
   * Initialize the statistic given a tree node that this statistic belongs to.
   * In this case, we ignore the node.
   */
  template<typename TreeType>
  RangeSearchStat(TreeType& /* node */) :
      lastDistance(0.0) { }

  //! Get the last distance evaluation.
  double LastDistance() const { return lastDistance; }
  //! Modify the last distance evaluation.
  double& LastDistance() { return lastDistance; }

  //! Serialize the statistic.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(lastDistance));
  }

 private:
  //! The last distance evaluation.
  double lastDistance;
};

} // namespace mlpack

#endif
