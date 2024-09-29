/**
 * @file core/tree/cover_tree/cover_tree.hpp
 * @author Ryan Curtin
 *
 * Definition of CoverTree, which can be used in place of the BinarySpaceTree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_HPP
#define MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/range.hpp>

#include "../statistic.hpp"
#include "first_point_is_root.hpp"

namespace mlpack {

/**
 * A cover tree is a tree specifically designed to speed up nearest-neighbor
 * computation in high-dimensional spaces.  Each non-leaf node references a
 * point and has a nonzero number of children, including a "self-child" which
 * references the same point.  A leaf node represents only one point.
 *
 * The tree can be thought of as a hierarchy with the root node at the top level
 * and the leaf nodes at the bottom level.  Each level in the tree has an
 * assigned 'scale' i.  The tree follows these two invariants:
 *
 * - nesting: the level C_i is a subset of the level C_{i - 1}.
 * - covering: all node in level C_{i - 1} have at least one node in the
 *     level C_i with distance less than or equal to b^i (exactly one of these
 *     is a parent of the point in level C_{i - 1}.
 *
 * Note that in the cover tree paper, there is a third invariant (the
 * 'separation invariant'), but that does not apply to our implementation,
 * because we have relaxed the invariant.
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
 * The CoverTree class offers three template parameters; a custom distance
 * metric type can be used with DistanceType (this class defaults to the
 * L2-squared metric).  The root node's point can be chosen with the
 * RootPointPolicy; by default, the FirstPointIsRoot policy is used, meaning the
 * first point in the dataset is used.  The StatisticType policy allows you to
 * define statistics which can be gathered during the creation of the tree.
 *
 * @tparam DistanceType Metric type to use during tree construction.
 * @tparam RootPointPolicy Determines which point to use as the root node.
 * @tparam StatisticType Statistic to be used during tree creation.
 * @tparam MatType Type of matrix to build the tree on (generally mat or
 *      sp_mat).
 */
template<typename DistanceType = LMetric<2, true>,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat,
         typename RootPointPolicy = FirstPointIsRoot>
class CoverTree
{
 public:
  //! So that other classes can access the matrix type.
  typedef MatType Mat;
  //! The type held by the matrix type.
  typedef typename MatType::elem_type ElemType;

  /**
   * Create the cover tree with the given dataset and given base.
   * The dataset will not be modified during the building procedure (unlike
   * BinarySpaceTree).
   *
   * The last argument will be removed in mlpack 1.1.0 (see #274 and #273).
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param base Base to use during tree building (default 2.0).
   * @param distance Distance metric to use (default NULL).
   */
  CoverTree(const MatType& dataset,
            const ElemType base = 2.0,
            DistanceType* distance = NULL);

  /**
   * Create the cover tree with the given dataset and the given instantiated
   * distance metric.  Optionally, set the base.  The dataset will not be
   * modified during the building procedure (unlike BinarySpaceTree).
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param distance Instantiated distance metric to use during tree building.
   * @param base Base to use during tree building (default 2.0).
   */
  CoverTree(const MatType& dataset,
            DistanceType& distance,
            const ElemType base = 2.0);

  /**
   * Create the cover tree with the given dataset, taking ownership of the
   * dataset.  Optionally, set the base.
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param base Base to use during tree building (default 2.0).
   */
  CoverTree(MatType&& dataset,
            const ElemType base = 2.0);

  /**
   * Create the cover tree with the given dataset and the given instantiated
   * distance metric, taking ownership of the dataset.  Optionally, set the
   * base.
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param distance Instantiated distance metric to use during tree building.
   * @param base Base to use during tree building (default 2.0).
   */
  CoverTree(MatType&& dataset,
            DistanceType& distance,
            const ElemType base = 2.0);

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
   * @param distance Distance metric to use (default NULL).
   */
  CoverTree(const MatType& dataset,
            const ElemType base,
            const size_t pointIndex,
            const int scale,
            CoverTree* parent,
            const ElemType parentDistance,
            arma::Col<size_t>& indices,
            arma::vec& distances,
            size_t nearSetSize,
            size_t& farSetSize,
            size_t& usedSetSize,
            DistanceType& distance = NULL);

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
   * @param distance Instantiated distance metric (optional).
   */
  CoverTree(const MatType& dataset,
            const ElemType base,
            const size_t pointIndex,
            const int scale,
            CoverTree* parent,
            const ElemType parentDistance,
            const ElemType furthestDescendantDistance,
            DistanceType* distance = NULL);

  /**
   * Create a cover tree from another tree.  Be careful!  This may use a lot of
   * memory and take a lot of time.  This will also make a copy of the dataset.
   *
   * @param other Cover tree to copy from.
   */
  CoverTree(const CoverTree& other);

  /**
   * Move constructor for a Cover Tree, possess all the members of the given
   * tree.
   *
   * @param other Cover Tree to move.
   */
  CoverTree(CoverTree&& other);

  /**
   * Copy the given Cover Tree.
   *
   * @param other The tree to be copied.
   */
  CoverTree& operator=(const CoverTree& other);

  /**
   * Take ownership of the given Cover Tree.
   *
   * @param other The tree to take ownership of.
   */
  CoverTree& operator=(CoverTree&& other);

  /**
   * Create a cover tree from a cereal archive.
   */
  template<typename Archive>
  CoverTree(
      Archive& ar,
      const typename std::enable_if_t<cereal::is_loading<Archive>()>* = 0);

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

  template<typename RuleType>
  using BreadthFirstDualTreeTraverser = DualTreeTraverser<RuleType>;

  //! Get a reference to the dataset.
  const MatType& Dataset() const { return *dataset; }

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

  CoverTree*& ChildPtr(const size_t index) { return children[index]; }

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
  ElemType Base() const { return base; }
  //! Modify the base; don't do this, you'll break everything.
  ElemType& Base() { return base; }

  //! Get the statistic for this node.
  const StatisticType& Stat() const { return stat; }
  //! Modify the statistic for this node.
  StatisticType& Stat() { return stat; }

  /**
   * Return the index of the nearest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetNearestChild(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0);

  /**
   * Return the index of the furthest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetFurthestChild(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0);

  /**
   * Return the index of the nearest child node to the given query node.  If it
   * can't decide, it will return NumChildren() (invalid index).
   */
  size_t GetNearestChild(const CoverTree& queryNode);

  /**
   * Return the index of the furthest child node to the given query node.  If it
   * can't decide, it will return NumChildren() (invalid index).
   */
  size_t GetFurthestChild(const CoverTree& queryNode);

  //! Return the minimum distance to another node.
  ElemType MinDistance(const CoverTree& other) const;

  //! Return the minimum distance to another node given that the point-to-point
  //! distance has already been calculated.
  ElemType MinDistance(const CoverTree& other, const ElemType distance) const;

  //! Return the minimum distance to another point.
  ElemType MinDistance(const arma::vec& other) const;

  //! Return the minimum distance to another point given that the distance from
  //! the center to the point has already been calculated.
  ElemType MinDistance(const arma::vec& other, const ElemType distance) const;

  //! Return the maximum distance to another node.
  ElemType MaxDistance(const CoverTree& other) const;

  //! Return the maximum distance to another node given that the point-to-point
  //! distance has already been calculated.
  ElemType MaxDistance(const CoverTree& other, const ElemType distance) const;

  //! Return the maximum distance to another point.
  ElemType MaxDistance(const arma::vec& other) const;

  //! Return the maximum distance to another point given that the distance from
  //! the center to the point has already been calculated.
  ElemType MaxDistance(const arma::vec& other, const ElemType distance) const;

  //! Return the minimum and maximum distance to another node.
  RangeType<ElemType> RangeDistance(const CoverTree& other) const;

  //! Return the minimum and maximum distance to another node given that the
  //! point-to-point distance has already been calculated.
  RangeType<ElemType> RangeDistance(const CoverTree& other,
                                          const ElemType distance) const;

  //! Return the minimum and maximum distance to another point.
  RangeType<ElemType> RangeDistance(const arma::vec& other) const;

  //! Return the minimum and maximum distance to another point given that the
  //! point-to-point distance has already been calculated.
  RangeType<ElemType> RangeDistance(const arma::vec& other,
                                          const ElemType distance) const;

  //! Get the parent node.
  CoverTree* Parent() const { return parent; }
  //! Modify the parent node.
  CoverTree*& Parent() { return parent; }

  //! Get the distance to the parent.
  ElemType ParentDistance() const { return parentDistance; }
  //! Modify the distance to the parent.
  ElemType& ParentDistance() { return parentDistance; }

  //! Get the distance to the furthest point.  This is always 0 for cover trees.
  ElemType FurthestPointDistance() const { return 0.0; }

  //! Get the distance from the center of the node to the furthest descendant.
  ElemType FurthestDescendantDistance() const
  { return furthestDescendantDistance; }
  //! Modify the distance from the center of the node to the furthest
  //! descendant.
  ElemType& FurthestDescendantDistance() { return furthestDescendantDistance; }

  //! Get the minimum distance from the center to any bound edge (this is the
  //! same as furthestDescendantDistance).
  ElemType MinimumBoundDistance() const { return furthestDescendantDistance; }

  //! Get the center of the node and store it in the given vector.
  void Center(arma::vec& center) const
  {
    center = arma::vec(dataset->col(point));
  }

  //! Get the instantiated distance metric.
  [[deprecated("Will be removed in mlpack 5.0.0; use Distance()")]]
  DistanceType& Metric() const { return *distance; }

  //! Get the instantiated distance metric.
  DistanceType& Distance() const { return *distance; }

 private:
  //! Reference to the matrix which this tree is built on.
  const MatType* dataset;
  //! Index of the point in the matrix which this node represents.
  size_t point;
  //! The list of children; the first is the self-child.
  std::vector<CoverTree*> children;
  //! Scale level of the node.
  int scale;
  //! The base used to construct the tree.
  ElemType base;
  //! The instantiated statistic.
  StatisticType stat;
  //! The number of descendant points.
  size_t numDescendants;
  //! The parent node (NULL if this is the root of the tree).
  CoverTree* parent;
  //! Distance to the parent.
  ElemType parentDistance;
  //! Distance to the furthest descendant.
  ElemType furthestDescendantDistance;
  //! Whether or not we need to destroy the distance metric in the destructor.
  bool localDistance;
  //! If true, we own the dataset and need to destroy it in the destructor.
  bool localDataset;
  //! The distance metric used for this tree.
  DistanceType* distance;

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
                      const ElemType bound,
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
                     const ElemType bound,
                     const size_t nearSetSize,
                     const size_t pointSetSize);

  /**
   * Take a look at the last child (the most recently created one) and remove
   * any implicit nodes that have been created.
   */
  void RemoveNewImplicitNodes();

 protected:
  /**
   * A default constructor.  This is meant to only be used with
   * cereal, which is allowed with the friend declaration below.
   * This does not return a valid tree!  This method must be protected, so that
   * the serialization shim can work with the default constructor.
   */
  CoverTree();

  //! Friend access is given for the default constructor.
  friend class cereal::access;

 public:
  /**
   * Serialize the tree.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  size_t DistanceComps() const { return distanceComps; }
  size_t& DistanceComps() { return distanceComps; }

 private:
  size_t distanceComps;
};

} // namespace mlpack

// Include implementation.
#include "cover_tree_impl.hpp"

// Include the rest of the pieces, if necessary.
#include "../cover_tree.hpp"

#endif
