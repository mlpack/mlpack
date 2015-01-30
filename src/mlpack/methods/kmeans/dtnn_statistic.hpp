/**
 * @file dtnn_statistic.hpp
 * @author Ryan Curtin
 *
 * Statistic for dual-tree nearest neighbor search based k-means clustering.
 */
#ifndef __MLPACK_METHODS_KMEANS_DTNN_STATISTIC_HPP
#define __MLPACK_METHODS_KMEANS_DTNN_STATISTIC_HPP

#include <mlpack/methods/neighbor_search/neighbor_search_stat.hpp>

namespace mlpack {
namespace kmeans {

class DTNNStatistic : public
    neighbor::NeighborSearchStat<neighbor::NearestNeighborSort>
{
 public:
  DTNNStatistic() :
      pruned(false),
      iteration(0),
      neighbor::NeighborSearchStat<neighbor::NearestNeighborSort>()
  {
    // Nothing to do.
  }

  DTNNStatistic(TreeType& /* node */) :
      pruned(false),
      iteration(0),
      neighbor::NeighborSearchStat<neighbor::NearestNeighborSort>()
  {
    // Nothing to do.
  }

  bool Pruned() const { return pruned; }
  bool& Pruned() { return pruned; }

  size_t Iteration() const { return iteration; }
  size_t& Iteration() { return iteration; }

 private:
  bool pruned;
  size_t iteration;
};

} // namespace kmeans
} // namespace mlpack

#endif
