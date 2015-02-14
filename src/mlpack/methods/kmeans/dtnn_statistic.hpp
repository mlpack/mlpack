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
      upperBound(DBL_MAX),
      lowerBound(DBL_MAX),
      owner(size_t(-1)),
      pruned(size_t(-1)),
      centroid(),
      trueLeft(NULL),
      trueRight(NULL),
      trueParent(NULL)
  {
    // Nothing to do.
  }

  template<typename TreeType>
  DTNNStatistic(TreeType& node) :
      neighbor::NeighborSearchStat<neighbor::NearestNeighborSort>(),
      upperBound(DBL_MAX),
      lowerBound(DBL_MAX),
      owner(size_t(-1)),
      pruned(size_t(-1)),
      trueLeft((void*) &node.Child(0)),
      trueRight((void*) &node.Child(1)),
      trueParent((void*) node.Parent())
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

  double UpperBound() const { return upperBound; }
  double& UpperBound() { return upperBound; }

  double LowerBound() const { return lowerBound; }
  double& LowerBound() { return lowerBound; }

  const arma::vec& Centroid() const { return centroid; }
  arma::vec& Centroid() { return centroid; }

  size_t Pruned() const { return pruned; }
  size_t& Pruned() { return pruned; }

  size_t Owner() const { return owner; }
  size_t& Owner() { return owner; }

  const void* TrueLeft() const { return trueLeft; }
  void*& TrueLeft() { return trueLeft; }

  const void* TrueRight() const { return trueRight; }
  void*& TrueRight() { return trueRight; }

  const void* TrueParent() const { return trueParent; }
  void*& TrueParent() { return trueParent; }

  std::string ToString() const
  {
    std::ostringstream o;
    o << "DTNNStatistic [" << this << "]:\n";
    o << "  Upper bound: " << upperBound << ".\n";
    o << "  Lower bound: " << lowerBound << ".\n";
    o << "  Pruned: " << pruned << ".\n";
    o << "  Owner: " << owner << ".\n";
    return o.str();
  }

 private:
  double upperBound;
  double lowerBound;
  size_t owner;
  size_t pruned;
  arma::vec centroid;
  void* trueLeft;
  void* trueRight;
  void* trueParent;
};

} // namespace kmeans
} // namespace mlpack

#endif
