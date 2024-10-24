/**
 * @file core/tree/spill_tree/spill_tree.hpp
 *
 * Definition of generalized hybrid spill tree (SpillTree).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_SPILL_TREE_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_SPILL_TREE_HPP

#include <mlpack/prereqs.hpp>
#include "../space_split/midpoint_space_split.hpp"
#include "../statistic.hpp"

namespace mlpack {

/**
 * A hybrid spill tree is a variant of binary space trees in which the children
 * of a node can "spill over" each other, and contain shared datapoints.
 *
 * Two new separating planes lplane and rplane are defined, both of which are
 * parallel to the original decision boundary and at a distance tau from it.
 * The region between lplane and rplane is called "overlapping buffer".
 *
 * For each node, we first split the points considering the overlapping buffer.
 * If either of its children contains more than rho fraction of the total points
 * we undo the overlapping splitting. Instead a conventional partition is used.
 * In this way, we can ensure that each split reduces the number of points of a
 * node by at least a constant factor.
 *
 * This particular tree does not allow growth, so you cannot add or delete nodes
 * from it.  If you need to add or delete a node, the better procedure is to
 * rebuild the tree entirely.
 *
 * Three runtime parameters are required in the constructor:
 *  - maxLeafSize: Max leaf size to be used.
 *  - tau: Overlapping size.
 *  - rho: Balance threshold.
 *
 * For more information on spill trees, see
 *
 * @code
 * @inproceedings{
 *   author = {Ting Liu, Andrew W. Moore, Alexander Gray and Ke Yang},
 *   title = {An Investigation of Practical Approximate Nearest Neighbor
 *     Algorithms},
 *   booktitle = {Advances in Neural Information Processing Systems 17},
 *   year = {2005},
 *   pages = {825--832}
 * }
 * @endcode
 *
 * @tparam DistanceType The distance metric used for tree-building.
 * @tparam StatisticType Extra data contained in the node.  See statistic.hpp
 *     for the necessary skeleton interface.
 * @tparam MatType The dataset class.
 * @tparam HyperplaneType The splitting hyperplane class.
 * @tparam SplitType The class that partitions the dataset/points at a
 *     particular node into two parts. Its definition decides the way this split
 *     is done.
 */
template<typename DistanceType,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat,
         template<typename HyperplaneDistanceType>
            class HyperplaneType = AxisOrthogonalHyperplane,
         template<typename SplitDistanceType, typename SplitMatType>
            class SplitType = MidpointSpaceSplit>
class SpillTree
{
 public:
  //! So other classes can use TreeType::Mat.
  using Mat = MatType;
  //! The type of element held in MatType.
  using ElemType = typename MatType::elem_type;
  //! The bound type.
  using BoundType = typename HyperplaneType<DistanceType>::BoundType;

 private:
  //! The left child node.
  SpillTree* left;
  //! The right child node.
  SpillTree* right;
  //! The parent node (NULL if this is the root of the tree).
  SpillTree* parent;
  //! The number of points of the dataset contained in this node (and its
  //! children).
  size_t count;
  //! The list of indexes of points contained in this node (non-NULL if the node
  //! is a leaf or if overlappingNode is true).
  arma::Col<size_t>* pointsIndex;
  //! Flag to distinguish overlapping nodes from non-overlapping nodes.
  bool overlappingNode;
  //! Splitting hyperplane represented by this node.
  HyperplaneType<DistanceType> hyperplane;
  //! The bound object for this node.
  BoundType bound;
  //! Any extra data contained in the node.
  StatisticType stat;
  //! The distance from the centroid of this node to the centroid of the parent.
  ElemType parentDistance;
  //! The worst possible distance to the furthest descendant, cached to speed
  //! things up.
  ElemType furthestDescendantDistance;
  //! The minimum distance from the center to any edge of the bound.
  ElemType minimumBoundDistance;
  //! The dataset.  If we are the root of the tree, we own the dataset and must
  //! delete it.
  const MatType* dataset;
  //! If true, we own the dataset and need to destroy it in the destructor.
  bool localDataset;

  //! A generic single-tree traverser for hybrid spill trees; see
  //! spill_single_tree_traverser.hpp for implementation.  The Defeatist
  //! template parameter determines if the traverser must do defeatist search on
  //! overlapping nodes.
  template<typename RuleType, bool Defeatist = false>
  class SpillSingleTreeTraverser;

  //! A generic dual-tree traverser for hybrid spill trees; see
  //! spill_dual_tree_traverser.hpp for implementation.  The Defeatist
  //! template parameter determines if the traverser must do defeatist search on
  //! overlapping nodes.
  template<typename RuleType, bool Defeatist = false>
  class SpillDualTreeTraverser;

 public:
  //! A single-tree traverser for hybrid spill trees.
  template<typename RuleType>
  using SingleTreeTraverser = SpillSingleTreeTraverser<RuleType, false>;

  //! A defeatist single-tree traverser for hybrid spill trees.
  template<typename RuleType>
  using DefeatistSingleTreeTraverser = SpillSingleTreeTraverser<RuleType, true>;

  //! A dual-tree traverser for hybrid spill trees.
  template<typename RuleType>
  using DualTreeTraverser = SpillDualTreeTraverser<RuleType, false>;

  //! A defeatist dual-tree traverser for hybrid spill trees.
  template<typename RuleType>
  using DefeatistDualTreeTraverser = SpillDualTreeTraverser<RuleType, true>;

  /**
   * Construct this as the root node of a hybrid spill tree using the given
   * dataset.  The dataset will not be modified during the building procedure
   * (unlike BinarySpaceTree).
   *
   * @param data Dataset to create tree from.
   * @param tau Overlapping size.
   * @param maxLeafSize Size of each leaf in the tree.
   * @param rho Balance threshold.
   */
  SpillTree(const MatType& data,
            const double tau = 0,
            const size_t maxLeafSize = 20,
            const double rho = 0.7);

  /**
   * Construct this as the root node of a hybrid spill tree using the given
   * dataset.  This will take ownership of the data matrix; if you don't want
   * this, consider using the constructor that takes a const reference to a
   * dataset.
   *
   * @param data Dataset to create tree from.
   * @param tau Overlapping size.
   * @param maxLeafSize Size of each leaf in the tree.
   * @param rho Balance threshold.
   */
  SpillTree(MatType&& data,
            const double tau = 0,
            const size_t maxLeafSize = 20,
            const double rho = 0.7);

  /**
   * Construct this node as a child of the given parent, including the given
   * list of points.  This is used for recursive tree-building by the other
   * constructors which don't specify point indices.
   *
   * @param parent Parent of this node.
   * @param points Vector of indexes of points to be included in this node.
   * @param tau Overlapping size.
   * @param maxLeafSize Size of each leaf in the tree.
   * @param rho Balance threshold.
   */
  SpillTree(SpillTree* parent,
            arma::Col<size_t>& points,
            const double tau = 0,
            const size_t maxLeafSize = 20,
            const double rho = 0.7);

  /**
   * Create a hybrid spill tree by copying the other tree.  Be careful!  This
   * can take a long time and use a lot of memory.
   *
   * @param other tree to be replicated.
   */
  SpillTree(const SpillTree& other);

  /**
   * Move constructor for a SpillTree; possess all the members of the given
   * tree.
   *
   * @param other tree to be moved.
   */
  SpillTree(SpillTree&& other);

   /**
   * Copy the given Spill Tree.
   *
   * @param other The tree to be copied.
   */
  SpillTree& operator=(const SpillTree& other);

  /**
   * Take ownership of the given Spill Tree.
   *
   * @param other The tree to take ownership of.
   */
  SpillTree& operator=(SpillTree&& other);

  /**
   * Initialize the tree from a cereal archive.
   *
   * @param ar Archive to load tree from.  Must be an iarchive, not an oarchive.
   */
  template<typename Archive>
  SpillTree(
      Archive& ar,
      const typename std::enable_if_t<cereal::is_loading<Archive>()>* = 0);

  /**
   * Deletes this node, deallocating the memory for the children and calling
   * their destructors in turn.  This will invalidate any pointers or references
   * to any nodes which are children of this one.
   */
  ~SpillTree();

  //! Return the bound object for this node.
  const BoundType& Bound() const { return bound; }
  //! Return the bound object for this node.
  BoundType& Bound() { return bound; }

  //! Return the statistic object for this node.
  const StatisticType& Stat() const { return stat; }
  //! Return the statistic object for this node.
  StatisticType& Stat() { return stat; }

  //! Return whether or not this node is a leaf (true if it has no children).
  bool IsLeaf() const;

  //! Gets the left child of this node.
  SpillTree* Left() const { return left; }
  //! Modify the left child of this node.
  SpillTree*& Left() { return left; }

  //! Gets the right child of this node.
  SpillTree* Right() const { return right; }
  //! Modify the right child of this node.
  SpillTree*& Right() { return right; }

  //! Gets the parent of this node.
  SpillTree* Parent() const { return parent; }
  //! Modify the parent of this node.
  SpillTree*& Parent() { return parent; }

  //! Get the dataset which the tree is built on.
  const MatType& Dataset() const { return *dataset; }

  //! Distinguish overlapping nodes from non-overlapping nodes.
  bool Overlap() const { return overlappingNode; }

  //! Get the Hyperplane instance.
  const HyperplaneType<DistanceType>& Hyperplane() const { return hyperplane; }

  //! Get the distance metric that the tree uses.
  [[deprecated("Will be removed in mlpack 5.0.0; use Distance()")]]
  DistanceType Metric() const { return DistanceType(); }

  //! Get the distance metric that the tree uses.
  DistanceType Distance() const { return DistanceType(); }

  //! Return the number of children in this node.
  size_t NumChildren() const;

  /**
   * Return the index of the nearest child node to the given query point (this
   * is an efficient estimation based on the splitting hyperplane, the node
   * returned is not necessarily the nearest).  If this is a leaf node, it will
   * return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetNearestChild(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0);

  /**
   * Return the index of the furthest child node to the given query point (this
   * is an efficient estimation based on the splitting hyperplane, the node
   * returned is not necessarily the furthest).  If this is a leaf node, it will
   * return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetFurthestChild(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0);

  /**
   * Return the index of the nearest child node to the given query node (this
   * is an efficient estimation based on the splitting hyperplane, the node
   * returned is not necessarily the nearest).  If it can't decide it will
   * return NumChildren() (invalid index).
   */
  size_t GetNearestChild(const SpillTree& queryNode);

  /**
   * Return the index of the furthest child node to the given query node (this
   * is an efficient estimation based on the splitting hyperplane, the node
   * returned is not necessarily the furthest).  If it can't decide it will
   * return NumChildren() (invalid index).
   */
  size_t GetFurthestChild(const SpillTree& queryNode);

  /**
   * Return the furthest distance to a point held in this node.  If this is not
   * a leaf node, then the distance is 0 because the node holds no points.
   */
  ElemType FurthestPointDistance() const;

  /**
   * Return the furthest possible descendant distance.  This returns the maximum
   * distance from the centroid to the edge of the bound and not the empirical
   * quantity which is the actual furthest descendant distance.  So the actual
   * furthest descendant distance may be less than what this method returns (but
   * it will never be greater than this).
   */
  ElemType FurthestDescendantDistance() const;

  //! Return the minimum distance from the center of the node to any bound edge.
  ElemType MinimumBoundDistance() const;

  //! Return the distance from the center of this node to the center of the
  //! parent node.
  ElemType ParentDistance() const { return parentDistance; }
  //! Modify the distance from the center of this node to the center of the
  //! parent node.
  ElemType& ParentDistance() { return parentDistance; }

  /**
   * Return the specified child (0 will be left, 1 will be right).  If the index
   * is greater than 1, this will return the right child.
   *
   * @param child Index of child to return.
   */
  SpillTree& Child(const size_t child) const;

  SpillTree*& ChildPtr(const size_t child)
  { return (child == 0) ? left : right; }

  //! Return the number of points in this node (0 if not a leaf).
  size_t NumPoints() const;

  /**
   * Return the number of descendants of this node.  For a non-leaf spill tree,
   * this is the number of points at the descendant leaves.  For a leaf, this is
   * the number of points in the leaf.
   */
  size_t NumDescendants() const;

  /**
   * Return the index (with reference to the dataset) of a particular descendant
   * of this node.  The index should be greater than zero but less than the
   * number of descendants.
   *
   * @param index Index of the descendant.
   */
  size_t Descendant(const size_t index) const;

  /**
   * Return the index (with reference to the dataset) of a particular point in
   * this node.  This will happily return invalid indices if the given index is
   * greater than the number of points in this node (obtained with NumPoints())
   * -- be careful.
   *
   * @param index Index of point for which a dataset index is wanted.
   */
  size_t Point(const size_t index) const;

  //! Return the minimum distance to another node.
  ElemType MinDistance(const SpillTree& other) const
  {
    return bound.MinDistance(other.Bound());
  }

  //! Return the maximum distance to another node.
  ElemType MaxDistance(const SpillTree& other) const
  {
    return bound.MaxDistance(other.Bound());
  }

  //! Return the minimum and maximum distance to another node.
  RangeType<ElemType> RangeDistance(const SpillTree& other) const
  {
    return bound.RangeDistance(other.Bound());
  }

  //! Return the minimum distance to another point.
  template<typename VecType>
  ElemType MinDistance(const VecType& point,
                       typename std::enable_if_t<IsVector<VecType>::value>* = 0)
      const
  {
    return bound.MinDistance(point);
  }

  //! Return the maximum distance to another point.
  template<typename VecType>
  ElemType MaxDistance(const VecType& point,
                       typename std::enable_if_t<IsVector<VecType>::value>* = 0)
      const
  {
    return bound.MaxDistance(point);
  }

  //! Return the minimum and maximum distance to another point.
  template<typename VecType>
  RangeType<ElemType>
  RangeDistance(const VecType& point,
                typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
  {
    return bound.RangeDistance(point);
  }

  //! Returns false: this tree type does not have self children.
  static bool HasSelfChildren() { return false; }

  //! Store the center of the bounding region in the given vector.
  void Center(arma::vec& center) { bound.Center(center); }

 private:
  /**
   * Splits the current node, assigning its left and right children recursively.
   *
   * @param points Vector of indexes of points to be included in this node.
   * @param maxLeafSize Maximum number of points held in a leaf.
   * @param tau Overlapping size.
   * @param rho Balance threshold.
   */
  void SplitNode(arma::Col<size_t>& points,
                 const size_t maxLeafSize,
                 const double tau,
                 const double rho);

  /**
   * Split the list of points.
   *
   * @param tau Overlapping size.
   * @param rho Balance threshold.
   * @param points Vector of indexes of points to be included.
   * @param leftPoints Indexes of points to be included in left child.
   * @param rightPoints Indexes of points to be included in right child.
   * @return Flag to know if the overlapping buffer was included.
   */
  bool SplitPoints(const double tau,
                   const double rho,
                   const arma::Col<size_t>& points,
                   arma::Col<size_t>& leftPoints,
                   arma::Col<size_t>& rightPoints);
 protected:
  /**
   * A default constructor.  This is meant to only be used with
   * cereal, which is allowed with the friend declaration below.
   * This does not return a valid tree!  The method must be protected, so that
   * the serialization shim can work with the default constructor.
   */
  SpillTree();

  //! Friend access is given for the default constructor.
  friend class cereal::access;

 public:
  /**
   * Serialize the tree.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);
};

} // namespace mlpack

// Include implementation.
#include "spill_tree_impl.hpp"

// Include everything else, if necessary.
#include "../spill_tree.hpp"

#endif
