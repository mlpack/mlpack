/**
 * @file pelleg_moore_kmeans_statistic.hpp
 * @author Ryan Curtin
 *
 * A StatisticType for trees which holds the blacklist for various k-means
 * clusters.  See the Pelleg and Moore paper for more details.
 */
#ifndef __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_STATISTIC_HPP
#define __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_STATISTIC_HPP

namespace mlpack {
namespace kmeans {

/**
 * A statistic for trees which holds the blacklist for Pelleg-Moore k-means
 * clustering (which represents the clusters that cannot possibly own any points
 * in a node).
 */
class PellegMooreKMeansStatistic
{
 public:
  //! Initialize the statistic without a node (this does nothing).
  PellegMooreKMeansStatistic() { }

  //! Initialize the statistic for a node; this calculates the centroid and
  //! caches it.
  template<typename TreeType>
  PellegMooreKMeansStatistic(TreeType& node)
  {
    centroid.zeros(node.Dataset().n_rows);

    // Hope it's a depth-first build procedure.  Also, this won't work right for
    // trees that have self-children or stuff like that.
    for (size_t i = 0; i < node.NumChildren(); ++i)
    {
      centroid += node.Child(i).NumDescendants() *
          node.Child(i).Stat().Centroid();
    }

    for (size_t i = 0; i < node.NumPoints(); ++i)
    {
      centroid += node.Dataset().col(node.Point(i));
    }

    if (node.NumDescendants() > 0)
      centroid /= node.NumDescendants();
    else
      centroid.fill(DBL_MAX); // Invalid centroid.  What else can we do?
  }

  //! Get the cluster blacklist.
  const arma::uvec& Blacklist() const { return blacklist; }
  //! Modify the cluster blacklist.
  arma::uvec& Blacklist() { return blacklist; }

  //! Get the node's centroid.
  const arma::vec& Centroid() const { return centroid; }
  //! Modify the node's centroid (be careful!).
  arma::vec& Centroid() { return centroid; }

  //! Return the object as a string.
  std::string ToString() const
  {
    std::ostringstream convert;
    convert << "KMeansStatistic [" << this << "]" << std::endl;
    convert << "  Blacklist: " << blacklist.t();
    convert << "  Centroid: " << centroid.t();
    return convert.str();
  }

 private:
  //! The cluster blacklist for the node.
  arma::uvec blacklist;
  //! The centroid of the node, cached for use during prunes.
  arma::vec centroid;
};

}; // namespace kmeans
}; // namespace mlpack

#endif
