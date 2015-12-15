/**
 * @file range_search_stat.hpp
 * @author Ryan Curtin
 *
 * Statistic class for RangeSearch, which just holds the last visited node and
 * the corresponding base case result.
 */
#ifndef __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_STAT_HPP
#define __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_STAT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace range {

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
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(lastDistance, "lastDistance");
  }

 private:
  //! The last distance evaluation.
  double lastDistance;
};

} // namespace neighbor
} // namespace mlpack

#endif
