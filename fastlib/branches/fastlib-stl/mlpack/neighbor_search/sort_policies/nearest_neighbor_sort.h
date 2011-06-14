/***
 * @file nearest_neighbor_sort.h
 * @author Ryan Curtin
 * 
 * Implementation of the SortPolicy class for NeighborSearch; in this case, the
 * nearest neighbors are those that are most important.
 */
#ifndef __MLPACK_NEIGHBOR_NEAREST_NEIGHBOR_SORT_H
#define __MLPACK_NEIGHBOR_NEAREST_NEIGHBOR_SORT_H

#include <fastlib/fastlib.h>
#include <armadillo>

namespace mlpack {
namespace neighbor {

class NearestNeighborSort {
 public:
  static index_t SortDistance(arma::vec& list, double new_distance);
};

}; // namespace neighbor
}; // namespace mlpack

#endif
