/**
 * @file cover_tree.hpp
 * @author Ryan Curtin
 *
 * Definition of CoverTree, which can be used in place of the BinarySpaceTree.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include "first_point_is_root.hpp"
#include "../statistic.hpp"

namespace mlpack {
namespace tree {

/**
 * A cover tree is a tree specifically designed to speed up nearest-neighbor
 * computation in high-dimensional spaces.  Each non-leaf node references a
 * point and has a nonzero number of children, including a "self-child" which
 * references the same point.  A leaf node represents only one point.
 *
 * The tree can be thought of as a hierarchy with the root node at the top level
 * and the leaf nodes at the bottom level.  Each level in the tree has an
 * assigned 'scale' i.  The tree follows these three conditions:
 *
 * - nesting: the level C_i is a subset of the level C_{i - 1}.
 * - covering: all node in level C_{i - 1} have at least one node in the
 *     level C_i with distance less than or equal to b^i (exactly one of these
 *     is a parent of the point in level C_{i - 1}.
 * - separation: all nodes in level C_i have distance greater than b^i to all
 *     other nodes in level C_i.
 *
 * The value 'b' refers to the base, which is a parameter of the tree.  These
 * three properties make the cover tree very good for fast, high-dimensional
 * nearest-neighbor search.
 *
 * The theoretical structure of the tree contains many 'implicit' nodes which
 * only have a "self-child" (a child referencing the same point, but at a lower
 * scale level).  This practical implementation only constructs explicit nodes
 * -- non-leaf nodes with more than one child.  A leaf node has no children, and
 * its scale level is INT_MIN.
 *
 * For more information on cover trees, see
 *
 * @code
 * @inproceedings{
 *   author = {Beygelzimer, Alina and Kakade, Sham and Langford, John},
 *   title = {Cover trees for nearest neighbor},
 *   booktitle = {Proceedings of the 23rd International Conference on Machine
 *     Learning},
 *   series = {ICML '06},
 *   year = {2006},
 *   pages = {97--104]
 * }
 * @endcode
 *
 * For information on runtime bounds of the nearest-neighbor computation using
 * cover trees, see the following paper, presented at NIPS 2009:
 *
 * @code
 * @inproceedings{
 *   author = {Ram, P., and Lee, D., and March, W.B., and Gray, A.G.},
 *   title = {Linear-time Algorithms for Pairwise Statistical Problems},
 *   booktitle = {Advances in Neural Information Processing Systems 22},
 *   editor = {Y. Bengio and D. Schuurmans and J. Lafferty and C.K.I. Williams
 *     and A. Culotta},
 *   pages = {1527--1535},
 *   year = {2009}
 * }
 * @endcode
 *
 * The CoverTree class offers three template parameters; a custom metric type
 * can be used with MetricType (this class defaults to the L2-squared metric).
 * The root node's point can be chosen with the RootPointPolicy; by default, the
 * FirstPointIsRoot policy is used, meaning the first point in the dataset is
 * used.  The StatisticType policy allows you to define statistics which can be
 * gathered during the creation of the tree.
 *
 * @tparam MetricType Metric type to use during tree construction.
 * @tparam RootPointPolicy Determines which point to use as the root node.
 * @tparam StatisticType Statistic to be used during tree creation.
 */
template<typename MetricType = metric::LMetric<2, true>,
         typename RootPointPolicy = FirstPointIsRoot,
         typename StatisticType = EmptyStatistic>
class CoverTree
{
 public:
  typedef arma::mat Mat;

  /**
   * Create the cover tree with the given dataset and given base.
   * The dataset will not be modified during the building procedure (unlike
   * BinarySpaceTree).
   *
   * The last argument will be removed in mlpack 1.1.0 (see #274 and #273).
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param base Base to use during tree building (default 2.0).
   */
  CoverTree(const arma::mat& dataset,
            const double base = 2.0,
            MetricType* metric = NULL);

  /**
   * Create the cover tree with the given dataset and the given instantiated
   * metric.  Optionally, set the base.  The dataset will not be modified during
   * the building procedure (unlike BinarySpaceTree).
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param metric Instantiated metric to use during tree building.
   * @param base Base to use during tree building (default 2.0).
   */
  CoverTree(const arma::mat& dataset,
            MetricType& metric,
            const double base = 2.0);

  /**
   * Construct a child cover tree node.  This constructor is not meant to be
   * used externally, but it could be used to insert another node into a tree.
   * This procedure uses only one vector for the near set, the far set, and the
   * used set (this is to prevent unnecessary memory allocation in recursive
   * calls to this constructor).  Therefore, the size of the near set, far set,
   * and used set must be passed in.  The near set will be entirely used up, and
   * some of the far set may be used.  The value of usedSetSize will be set to
   * the number of points used in the construction of this node, and the value
   * of farSetSize will be modified to reflect the number of points in the far
   * set _after_ the construction of this node.
   *
   * If you are calling this manually, be careful that the given scale is
   * as small as possible, or you may be creating an implicit node in your tree.
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param base Base to use during tree building.
   * @param pointIndex Index of the point this node references.
   * @param scale Scale of this level in the tree.
   * @param parent Parent of this node (NULL indicates no parent).
   * @param parentDistance Distance to the parent node.
   * @param indices Array of indices, ordered [ nearSet | farSet | usedSet ];
   *     will be modified to [ farSet | usedSet ].
   * @param distances Array of distances, ordered the same way as the indices.
   *     These represent the distances between the point specified by pointIndex
   *     and each point in the indices array.
   * @param nearSetSize Size of the near set; if 0, this will be a leaf.
   * @param farSetSize Size of the far set; may be modified (if this node uses
   *     any points in the far set).
   * @param usedSetSize The number of points used will be added to this number.
   */
  CoverTree(const arma::mat& dataset,
            const double base,
            const size_t pointIndex,
            const int scale,
            CoverTree* parent,
            const double parentDistance,
            arma::Col<size_t>& indices,
            arma::vec& distances,
            size_t nearSetSize,
            size_t& farSetSize,
            size_t& usedSetSize,
            MetricType& metric = NULL);

  /**
   * Manually construct a cover tree node; no tree assembly is done in this
   * constructor, and children must be added manually (use Children()).  This
   * constructor is useful when the tree is being "imported" into the CoverTree
   * class after being created in some other manner.
   *
   * @param dataset Reference to the dataset this node is a part of.
   * @param base Base that was used for tree building.
   * @param pointIndex Index of the point in the dataset which this node refers
   *      to.
   * @param scale Scale of this node's level in the tree.
   * @param parent Parent node (NULL indicates no parent).
   * @param parentDistance Distance to parent node point.
   * @param furthestDescendantDistance Distance to furthest descendant point.
   * @param metric Instantiated metric (optional).
   */
  CoverTree(const arma::mat& dataset,
            const double base,
            const size_t pointIndex,
            const int scale,
            CoverTree* parent,
            const double parentDistance,
            const double furthestDescendantDistance,
            MetricType* metric = NULL);

  /**
   * Create a cover tree from another tree.  Be careful!  This may use a lot of
   * memory and take a lot of time.
   *
   * @param other Cover tree to copy from.
   */
  CoverTree(const CoverTree& other);

  /**
   * Delete this cover tree node and its children.
   */
  ~CoverTree();

  //! A single-tree cover tree traverser; see single_tree_traverser.hpp for
  //! implementation.
  template<typename RuleType>
  class SingleTreeTraverser;

  //! A dual-tree cover tree traverser; see dual_tree_traverser.hpp.
  template<typename RuleType>
  class DualTreeTraverser;

  //! Get a reference to the dataset.
  const arma::mat& Dataset() const { return dataset; }

  //! Get the index of the point which this node represents.
  size_t Point() const { return point; }
  //! For compatibility with other trees; the argument is ignored.
  size_t Point(const size_t) const { return point; }

  bool IsLeaf() const { return (children.size() == 0); }
  size_t NumPoints() const { return 1; }

  //! Get a particular child node.
  const CoverTree& Child(const size_t index) const { return *children[index]; }
  //! Modify a particular child node.
  CoverTree& Child(const size_t index) { return *children[index]; }

  //! Get the number of children.
  size_t NumChildren() const { return children.size(); }

  //! Get the children.
  const std::vector<CoverTree*>& Children() const { return children; }
  //! Modify the children manually (maybe not a great idea).
  std::vector<CoverTree*>& Children() { return children; }

  //! Get the number of descendant points.
  size_t NumDescendants() const;

  //! Get the index of a particular descendant point.
  size_t Descendant(const size_t index) const;

  //! Get the scale of this node.
  int Scale() const { return scale; }
  //! Modify the scale of this node.  Be careful...
  int& Scale() { return scale; }

  //! Get the base.
  double Base() const { return base; }
  //! Modify the base; don't do this, you'll break everything.
  double& Base() { return base; }

  //! Get the statistic for this node.
  const StatisticType& Stat() const { return stat; }
  //! Modify the statistic for this node.
  StatisticType& Stat() { return stat; }

  //! Return the minimum distance to another node.
  double MinDistance(const CoverTree* other) const;

  //! Return the minimum distance to another node given that the point-to-point
  //! distance has already been calculated.
  double MinDistance(const CoverTree* other, const double distance) const;

  //! Return the minimum distance to another point.
  double MinDistance(const arma::vec& other) const;

  //! Return the minimum distance to another point given that the distance from
  //! the center to the point has already been calculated.
  double MinDistance(const arma::vec& other, const double distance) const;

  //! Return the maximum distance to another node.
  double MaxDistance(const CoverTree* other) const;

  //! Return the maximum distance to another node given that the point-to-point
  //! distance has already been calculated.
  double MaxDistance(const CoverTree* other, const double distance) const;

  //! Return the maximum distance to another point.
  double MaxDistance(const arma::vec& other) const;

  //! Return the maximum distance to another point given that the distance from
  //! the center to the point has already been calculated.
  double MaxDistance(const arma::vec& other, const double distance) const;

  //! Return the minimum and maximum distance to another node.
  math::Range RangeDistance(const CoverTree* other) const;

  //! Return the minimum and maximum distance to another node given that the
  //! point-to-point distance has already been calculated.
  math::Range RangeDistance(const CoverTree* other, const double distance)
      const;

  //! Return the minimum and maximum distance to another point.
  math::Range RangeDistance(const arma::vec& other) const;

  //! Return the minimum and maximum distance to another point given that the
  //! point-to-point distance has already been calculated.
  math::Range RangeDistance(const arma::vec& other, const double distance)
      const;

  //! Returns true: this tree does have self-children.
  static bool HasSelfChildren() { return true; }

  //! Get the parent node.
  CoverTree* Parent() const { return parent; }
  //! Modify the parent node.
  CoverTree*& Parent() { return parent; }

  //! Get the distance to the parent.
  double ParentDistance() const { return parentDistance; }
  //! Modify the distance to the parent.
  double& ParentDistance() { return parentDistance; }

  //! Get the distance to the furthest point.  This is always 0 for cover trees.
  double FurthestPointDistance() const { return 0.0; }

  //! Get the distance from the center of the node to the furthest descendant.
  double FurthestDescendantDistance() const
  { return furthestDescendantDistance; }
  //! Modify the distance from the center of the node to the furthest
  //! descendant.
  double& FurthestDescendantDistance() { return furthestDescendantDistance; }

  //! Get the minimum distance from the center to any bound edge (this is the
  //! same as furthestDescendantDistance).
  double MinimumBoundDistance() const { return furthestDescendantDistance; }

  //! Get the centroid of the node and store it in the given vector.
  void Centroid(arma::vec& centroid) const { centroid = dataset.col(point); }

  //! Get the instantiated metric.
  MetricType& Metric() const { return *metric; }

 private:
  //! Reference to the matrix which this tree is built on.
  const arma::mat& dataset;

  //! Index of the point in the matrix which this node represents.
  size_t point;

  //! The list of children; the first is the self-child.
  std::vector<CoverTree*> children;

  //! Scale level of the node.
  int scale;

  //! The base used to construct the tree.
  double base;

  //! The instantiated statistic.
  StatisticType stat;

  //! The number of descendant points.
  size_t numDescendants;

  //! The parent node (NULL if this is the root of the tree).
  CoverTree* parent;

  //! Distance to the parent.
  double parentDistance;

  //! Distance to the furthest descendant.
  double furthestDescendantDistance;

  //! Whether or not we need to destroy the metric in the destructor.
  bool localMetric;

  //! The metric used for this tree.
  MetricType* metric;

  /**
   * Create the children for this node.
   */
  void CreateChildren(arma::Col<size_t>& indices,
                      arma::vec& distances,
                      size_t nearSetSize,
                      size_t& farSetSize,
                      size_t& usedSetSize);

  /**
   * Fill the vector of distances with the distances between the point specified
   * by pointIndex and each point in the indices array.  The distances of the
   * first pointSetSize points in indices are calculated (so, this does not
   * necessarily need to use all of the points in the arrays).
   *
   * @param pointIndex Point to build the distances for.
   * @param indices List of indices to compute distances for.
   * @param distances Vector to store calculated distances in.
   * @param pointSetSize Number of points in arrays to calculate distances for.
   */
  void ComputeDistances(const size_t pointIndex,
                        const arma::Col<size_t>& indices,
                        arma::vec& distances,
                        const size_t pointSetSize);
  /**
   * Split the given indices and distances into a near and a far set, returning
   * the number of points in the near set.  The distances must already be
   * initialized.  This will order the indices and distances such that the
   * points in the near set make up the first part of the array and the far set
   * makes up the rest:  [ nearSet | farSet ].
   *
   * @param indices List of indices; will be reordered.
   * @param distances List of distances; will be reordered.
   * @param bound If the distance is less than or equal to this bound, the point
   *      is placed into the near set.
   * @param pointSetSize Size of point set (because we may be sorting a smaller
   *      list than the indices vector will hold).
   */
  size_t SplitNearFar(arma::Col<size_t>& indices,
                      arma::vec& distances,
                      const double bound,
                      const size_t pointSetSize);

  /**
   * Assuming that the list of indices and distances is sorted as
   * [ childFarSet | childUsedSet | farSet | usedSet ],
   * resort the sets so the organization is
   * [ childFarSet | farSet | childUsedSet | usedSet ].
   *
   * The size_t parameters specify the sizes of each set in the array.  Only the
   * ordering of the indices and distances arrays will be modified (not their
   * actual contents).
   *
   * The size of any of the four sets can be zero and this method will handle
   * that case accordingly.
   *
   * @param indices List of indices to sort.
   * @param distances List of distances to sort.
   * @param childFarSetSize Number of points in child far set (childFarSet).
   * @param childUsedSetSize Number of points in child used set (childUsedSet).
   * @param farSetSize Number of points in far set (farSet).
   */
  size_t SortPointSet(arma::Col<size_t>& indices,
                      arma::vec& distances,
                      const size_t childFarSetSize,
                      const size_t childUsedSetSize,
                      const size_t farSetSize);

  void MoveToUsedSet(arma::Col<size_t>& indices,
                     arma::vec& distances,
                     size_t& nearSetSize,
                     size_t& farSetSize,
                     size_t& usedSetSize,
                     arma::Col<size_t>& childIndices,
                     const size_t childFarSetSize,
                     const size_t childUsedSetSize);
  size_t PruneFarSet(arma::Col<size_t>& indices,
                     arma::vec& distances,
                     const double bound,
                     const size_t nearSetSize,
                     const size_t pointSetSize);

  /**
   * Take a look at the last child (the most recently created one) and remove
   * any implicit nodes that have been created.
   */
  void RemoveNewImplicitNodes();

 public:
  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;

  size_t DistanceComps() const { return distanceComps; }
  size_t& DistanceComps() { return distanceComps; }

 private:
  size_t distanceComps;
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "cover_tree_impl.hpp"

#endif
