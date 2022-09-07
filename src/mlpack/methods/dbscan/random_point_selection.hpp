/**
 * @file methods/dbscan/random_point_selection.hpp
 * @author Ryan Curtin
 *
 * Randomly select the next point for DBSCAN.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DBSCAN_RANDOM_POINT_SELECTION_HPP
#define MLPACK_METHODS_DBSCAN_RANDOM_POINT_SELECTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class can be used to randomly select the next point to use for DBSCAN.
 */
class RandomPointSelection
{
 public:
  /**
   * Select the next point to use, randomly.
   *
   * @param * (point) Unused data.
   * @param data Dataset to cluster.
   */
  template<typename MatType>
  size_t Select(const size_t /* point */,
                const MatType& data)
  {
    // Initialize the length of the unvisited bitset.
    size_t size = data.n_cols; // Get the size of points.
    if (unvisited.size() != size)
      unvisited.resize(size, true); // Resize & Set bitset to one.

    // Count the unvisited points and generate nth index randomly.
    const size_t max = std::count(unvisited.begin(), unvisited.end(), true);
    const size_t index = RandInt(max);

    // Select the index'th unvisited point.
    size_t found = 0;
    for (size_t i = 0; i < unvisited.size(); ++i)
    {
      if (unvisited[i])
        ++found;

      if (found > index)
      {
        unvisited[i].flip(); // Set unvisited point to visited point.
        return i;
      }
    }
    return 0; // Not sure if it is possible to get here.
  }

 private:
  // Bitset for unvisited points. If true, mean unvisited.
  std::vector<bool> unvisited;
};

} // namespace mlpack

#endif
