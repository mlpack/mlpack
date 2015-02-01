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
      neighbor::NeighborSearchStat<neighbor::NearestNeighborSort>(),
      pruned(false),
      iteration(0),
      maxClusterDistance(DBL_MAX),
      secondClusterBound(0.0),
      owner(size_t(-1)),
      centroid()
  {
    // Nothing to do.
  }

  template<typename TreeType>
  DTNNStatistic(TreeType& node) :
      neighbor::NeighborSearchStat<neighbor::NearestNeighborSort>(),
      pruned(false),
      iteration(0),
      maxClusterDistance(DBL_MAX),
      secondClusterBound(0.0),
      owner(size_t(-1))
  {
    // Empirically calculate the centroid.
    centroid.zeros(node.Dataset().n_rows);
    for (size_t i = 0; i < node.NumPoints(); ++i)
      centroid += node.Dataset().col(node.Point(i));

    for (size_t i = 0; i < node.NumChildren(); ++i)
      centroid += node.Child(i).NumDescendants() *
          node.Child(i).Stat().Centroid();

    centroid /= node.NumDescendants();
  }

  bool Pruned() const { return pruned; }
  bool& Pruned() { return pruned; }

  size_t Iteration() const { return iteration; }
  size_t& Iteration() { return iteration; }

  double MaxClusterDistance() const { return maxClusterDistance; }
  double& MaxClusterDistance() { return maxClusterDistance; }

  double SecondClusterBound() const { return secondClusterBound; }
  double& SecondClusterBound() { return secondClusterBound; }

  size_t Owner() const { return owner; }
  size_t& Owner() { return owner; }

  const arma::vec& Centroid() const { return centroid; }
  arma::vec& Centroid() { return centroid; }

  std::string ToString() const
  {
    std::ostringstream o;
    o << "DTNNStatistic [" << this << "]:\n";
    o << "  Pruned: " << pruned << ".\n";
    o << "  Iteration: " << iteration << ".\n";
    o << "  MaxClusterDistance: " << maxClusterDistance << ".\n";
    o << "  SecondClusterBound: " << secondClusterBound << ".\n";
    o << "  Owner: " << owner << ".\n";
    return o.str();
  }

 private:
  bool pruned;
  size_t iteration;
  double maxClusterDistance;
  double secondClusterBound;
  size_t owner;
  arma::vec centroid;
};

} // namespace kmeans
} // namespace mlpack

#endif
