/**
 * @file random_point_selection.hpp
 * @author Ryan Curtin
 *
 * Randomly select the next point for DBSCAN.
 */
#ifndef MLPACK_METHODS_DBSCAN_RANDOM_POINT_SELECTION_HPP
#define MLPACK_METHODS_DBSCAN_RANDOM_POINT_SELECTION_HPP

#include <mlpack/prereqs.hpp>
#include <boost/dynamic_bitset.hpp>

namespace mlpack {
namespace dbscan {

class RandomPointSelection
{
 public:
  /**
   * Select the next point to use, randomly.
   */
  template<typename MatType>
  static size_t Select(const boost::dynamic_bitset<>& unvisited,
                       const MatType& /* data */)
  {
    const size_t max = unvisited.count();
    const size_t index = math::RandInt(max);

    // Select the index'th unvisited point.
    size_t i = 0;
    size_t found = 0;
    for (size_t i = 0; i < unvisited.size(); ++i)
    {
      if (unvisited[i])
        ++found;

      if (found == index)
        break;
    }

    return i;
  }
};

} // namespace dbscan
} // namespace mlpack

#endif
