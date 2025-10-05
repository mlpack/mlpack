/**
 * @file methods/tsne/centroid_statistic.hpp
 * @author Ryan Curtin
 * @author Ranjodh Singh
 *
 * Definition and Implementation of the Centroid Statistic.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_CENTROID_STATISTIC
#define MLPACK_METHODS_TSNE_CENTROID_STATISTIC

#include <mlpack/prereqs.hpp>
#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {

/**
 * Statistic that stores the Centroid/Center Of Mass/Barycenter
 * of a metric tree node.
 *
 * This statistic is intended for use with metric tree nodes (for example,
 * mlpack::Octree). For leaf nodes, the center of mass is the mean of the
 * points contained in the leaf. For internal nodes, the centroid is computed
 * as the weighted mean of child centroids, where the weight is the number of
 * descendants in the corresponding child.
 *
 * @tparam VecType Vector type used to store the centroid (default is
 *         arma::vec).
 */
template <typename VecType = arma::vec>
class CentroidStatisticType
{
 public:
  /**
   * Default constructor for the CentroidStatisticType.
   */
  CentroidStatisticType() {}

  /**
   * Construct from a tree node.
   *
   * @param node Tree node whose centroid is to be computed.
   */
  template <typename TreeType>
  CentroidStatisticType(const TreeType& node)
  {
    centroid.zeros(node.Dataset().n_rows);
    for (size_t i = 0; i < node.NumPoints(); ++i)
    {
      // Correct handling of cover tree: don't double-count the point
      // which appears in the children. (See dual_tree_kmeans_statistic.hpp)
      if (TreeTraits<TreeType>::HasSelfChildren && i == 0 &&
          node.NumChildren() > 0)
        continue;
      centroid += node.Dataset().col(node.Point(i));
    }

    for (size_t i = 0; i < node.NumChildren(); i++)
      centroid += node.Child(i).NumDescendants() *
                  node.Child(i).Stat().Centroid();
    centroid /= node.NumDescendants();
  }

  //! Get the centroid.
  const VecType& Centroid() const { return centroid; }
  //! Modify the centroid.
  VecType& Centroid() { return centroid; }

 private:
  //! The centroid vector.
  VecType centroid;
};

using CentroidStatistic = CentroidStatisticType<>;

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_CENTROID_STATISTIC
