/**
 * @file binary_space_tree.hpp
 *
 * Definition of generalized binary space partitioning tree (BinarySpaceTree).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_BINARY_SPACE_TREE_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_BINARY_SPACE_TREE_HPP

#include <mlpack/core.hpp>

#include "../statistic.hpp"
#include "midpoint_split.hpp"

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A binary space partitioning tree, such as a KD-tree or a ball tree.  Once the
 * bound and type of dataset is defined, the tree will construct itself.  Call
 * the constructor with the dataset to build the tree on, and the entire tree
 * will be built.
 *
 * This particular tree does not allow growth, so you cannot add or delete nodes
 * from it.  If you need to add or delete a node, the better procedure is to
 * rebuild the tree entirely.
 *
 * This tree does take one runtime parameter in the constructor, which is the
 * max leaf size to be used.
 *
 * @tparam MetricType The metric used for tree-building.  The BoundType may
 *     place restrictions on the metrics that can be used.
 * @tparam StatisticType Extra data contained in the node.  See statistic.hpp
 *     for the necessary skeleton interface.
 * @tparam MatType The dataset class.
 * @tparam BoundType The bound used for each node.  HRectBound, the default,
 *     requires that an LMetric<> is used for MetricType (so, EuclideanDistance,
 *     ManhattanDistance, etc.).
 * @tparam SplitType The class that partitions the dataset/points at a
 *     particular node into two parts. Its definition decides the way this split
 *     is done.
 */
template<typename MetricType,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat,
         template<typename BoundMetricType, typename...> class BoundType =
            bound::HRectBound,
         template<typename SplitBoundType, typename SplitMatType>
            class SplitType = MidpointSplit>
class BinarySpaceTree
{
 public:
  //! So other classes can use TreeType::Mat.
  typedef MatType Mat;
  //! The type of element held in MatType.
  typedef typename MatType::elem_type ElemType;

  typedef SplitType<BoundType<MetricType>, MatType> Split;

 private:
  //! The left child node.
  BinarySpaceTree* left;
  //! The right child node.
  BinarySpaceTree* right;
  //! The parent node (NULL if this is the root of the tree).
  BinarySpaceTree* parent;
  //! The index of the first point in the dataset contained in this node (and
  //! its children).
  size_t begin;
  //! The number of points of the dataset contained in this node (and its
  //! children).
  size_t count;
  //! The bound object for this node.
  BoundType<MetricType> bound;
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
  MatType* dataset;

 public:
  //! A single-tree traverser for binary space trees; see
  //! single_tree_traverser.hpp for implementation.
  template<typename RuleType>
  class SingleTreeTraverser;

  //! A dual-tree traverser for binary space trees; see dual_tree_traverser.hpp.
  template<typename RuleType>
  class DualTreeTraverser;

  template<typename RuleType>
  class BreadthFirstDualTreeTraverser;

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will copy the input matrix; if you don't want this, consider
   * using the constructor that takes an rvalue reference and use std::move().
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param maxLeafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(const MatType& data, const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will copy the input matrix and modify its ordering; a
   * mapping of the old point indices to the new point indices is filled.  If
   * you don't want the matrix to be copied, consider using the constructor that
   * takes an rvalue reference and use std::move().
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param maxLeafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(const MatType& data,
                  std::vector<size_t>& oldFromNew,
                  const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will copy the input matrix and modify its ordering; a
   * mapping of the old point indices to the new point indices is filled, as
   * well as a mapping of the new point indices to the old point indices.  If
   * you don't want the matrix to be copied, consider using the constructor that
   * takes an rvalue reference and use std::move().
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param newFromOld Vector which will be filled with the new positions for
   *     each old point.
   * @param maxLeafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(const MatType& data,
                  std::vector<size_t>& oldFromNew,
                  std::vector<size_t>& newFromOld,
                  const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will take ownership of the data matrix; if you don't want
   * this, consider using the constructor that takes a const reference to a
   * dataset.
   *
   * @param data Dataset to create tree from.
   * @param maxLeafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(MatType&& data,
                  const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will take ownership of the data matrix; a mapping of the
   * old point indices to the new point indices is filled.  If you don't want
   * the matrix to have its ownership taken, consider using the constructor that
   * takes a const reference to a dataset.
   *
   * @param data Dataset to create tree from.
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param maxLeafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(MatType&& data,
                  std::vector<size_t>& oldFromNew,
                  const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will take ownership of the data matrix; a mapping of the old
   * point indices to the new point indices is filled, as well as a mapping of
   * the new point indices to the old point indices.  If you don't want the
   * matrix to have its ownership taken, consider using the constructor that
   * takes a const reference to a dataset.
   *
   * @param data Dataset to create tree from.
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param newFromOld Vector which will be filled with the new positions for
   *     each old point.
   * @param maxLeafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(MatType&& data,
                  std::vector<size_t>& oldFromNew,
                  std::vector<size_t>& newFromOld,
                  const size_t maxLeafSize = 20);

  /**
   * Construct this node as a child of the given parent, starting at column
   * begin and using count points.  The ordering of that subset of points in the
   * parent's data matrix will be modified!  This is used for recursive
   * tree-building by the other constructors which don't specify point indices.
   *
   * @param parent Parent of this node.  Its dataset will be modified!
   * @param begin Index of point to start tree construction with.
   * @param count Number of points to use to construct tree.
   * @param splitter Instantiated node splitter object.
   * @param maxLeafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(BinarySpaceTree* parent,
                  const size_t begin,
                  const size_t count,
                  SplitType<BoundType<MetricType>, MatType>& splitter,
                  const size_t maxLeafSize = 20);

  /**
   * Construct this node as a child of the given parent, starting at column
   * begin and using count points.  The ordering of that subset of points in the
   * parent's data matrix will be modified!  This is used for recursive
   * tree-building by the other constructors which don't specify point indices.
   *
   * A mapping of the old point indices to the new point indices is filled, but
   * it is expected that the vector is already allocated with size greater than
   * or equal to (begin + count), and if that is not true, invalid memory reads
   * (and writes) will occur.
   *
   * @param parent Parent of this node.  Its dataset will be modified!
   * @param begin Index of point to start tree construction with.
   * @param count Number of points to use to construct tree.
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param splitter Instantiated node splitter object.
   * @param maxLeafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(BinarySpaceTree* parent,
                  const size_t begin,
                  const size_t count,
                  std::vector<size_t>& oldFromNew,
                  SplitType<BoundType<MetricType>, MatType>& splitter,
                  const size_t maxLeafSize = 20);

  /**
   * Construct this node as a child of the given parent, starting at column
   * begin and using count points.  The ordering of that subset of points in the
   * parent's data matrix will be modified!  This is used for recursive
   * tree-building by the other constructors which don't specify point indices.
   *
   * A mapping of the old point indices to the new point indices is filled, as
   * well as a mapping of the new point indices to the old point indices.  It is
   * expected that the vector is already allocated with size greater than or
   * equal to (begin_in + count_in), and if that is not true, invalid memory
   * reads (and writes) will occur.
   *
   * @param parent Parent of this node.  Its dataset will be modified!
   * @param begin Index of point to start tree construction with.
   * @param count Number of points to use to construct tree.
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param newFromOld Vector which will be filled with the new positions for
   *     each old point.
   * @param maxLeafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(BinarySpaceTree* parent,
                  const size_t begin,
                  const size_t count,
                  std::vector<size_t>& oldFromNew,
                  std::vector<size_t>& newFromOld,
                  SplitType<BoundType<MetricType>, MatType>& splitter,
                  const size_t maxLeafSize = 20);

  /**
   * Create a binary space tree by copying the other tree.  Be careful!  This
   * can take a long time and use a lot of memory.
   *
   * @param other Tree to be replicated.
   */
  BinarySpaceTree(const BinarySpaceTree& other);

  /**
   * Move constructor for a BinarySpaceTree; possess all the members of the
   * given tree.
   */
  BinarySpaceTree(BinarySpaceTree&& other);

  /**
   * Initialize the tree from a boost::serialization archive.
   *
   * @param ar Archive to load tree from.  Must be an iarchive, not an oarchive.
   */
  template<typename Archive>
  BinarySpaceTree(
      Archive& ar,
      const typename boost::enable_if<typename Archive::is_loading>::type* = 0);

  /**
   * Deletes this node, deallocating the memory for the children and calling
   * their destructors in turn.  This will invalidate any pointers or references
   * to any nodes which are children of this one.
   */
  ~BinarySpaceTree();

  //! Return the bound object for this node.
  const BoundType<MetricType>& Bound() const { return bound; }
  //! Return the bound object for this node.
  BoundType<MetricType>& Bound() { return bound; }

  //! Return the statistic object for this node.
  const StatisticType& Stat() const { return stat; }
  //! Return the statistic object for this node.
  StatisticType& Stat() { return stat; }

  //! Return whether or not this node is a leaf (true if it has no children).
  bool IsLeaf() const;

  //! Gets the left child of this node.
  BinarySpaceTree* Left() const { return left; }
  //! Modify the left child of this node.
  BinarySpaceTree*& Left() { return left; }

  //! Gets the right child of this node.
  BinarySpaceTree* Right() const { return right; }
  //! Modify the right child of this node.
  BinarySpaceTree*& Right() { return right; }

  //! Gets the parent of this node.
  BinarySpaceTree* Parent() const { return parent; }
  //! Modify the parent of this node.
  BinarySpaceTree*& Parent() { return parent; }

  //! Get the dataset which the tree is built on.
  const MatType& Dataset() const { return *dataset; }
  //! Modify the dataset which the tree is built on.  Be careful!
  MatType& Dataset() { return *dataset; }

  //! Get the metric that the tree uses.
  MetricType Metric() const { return MetricType(); }

  //! Return the number of children in this node.
  size_t NumChildren() const;

  /**
   * Return the index of the nearest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetNearestChild(
      const VecType& point,
      typename boost::enable_if<IsVector<VecType> >::type* = 0);

  /**
   * Return the index of the furthest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetFurthestChild(
      const VecType& point,
      typename boost::enable_if<IsVector<VecType> >::type* = 0);

  /**
   * Return the index of the nearest child node to the given query node.  If it
   * can't decide, it will return NumChildren() (invalid index).
   */
  size_t GetNearestChild(const BinarySpaceTree& queryNode);

  /**
   * Return the index of the furthest child node to the given query node.  If it
   * can't decide, it will return NumChildren() (invalid index).
   */
  size_t GetFurthestChild(const BinarySpaceTree& queryNode);

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
  BinarySpaceTree& Child(const size_t child) const;

  BinarySpaceTree*& ChildPtr(const size_t child)
  { return (child == 0) ? left : right; }

  //! Return the number of points in this node (0 if not a leaf).
  size_t NumPoints() const;

  /**
   * Return the number of descendants of this node.  For a non-leaf in a binary
   * space tree, this is the number of points at the descendant leaves.  For a
   * leaf, this is the number of points in the leaf.
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
  ElemType MinDistance(const BinarySpaceTree& other) const
  {
    return bound.MinDistance(other.Bound());
  }

  //! Return the maximum distance to another node.
  ElemType MaxDistance(const BinarySpaceTree& other) const
  {
    return bound.MaxDistance(other.Bound());
  }

  //! Return the minimum and maximum distance to another node.
  math::RangeType<ElemType> RangeDistance(const BinarySpaceTree& other) const
  {
    return bound.RangeDistance(other.Bound());
  }

  //! Return the minimum distance to another point.
  template<typename VecType>
  ElemType MinDistance(const VecType& point,
                       typename boost::enable_if<IsVector<VecType> >::type* = 0)
      const
  {
    return bound.MinDistance(point);
  }

  //! Return the maximum distance to another point.
  template<typename VecType>
  ElemType MaxDistance(const VecType& point,
                       typename boost::enable_if<IsVector<VecType> >::type* = 0)
      const
  {
    return bound.MaxDistance(point);
  }

  //! Return the minimum and maximum distance to another point.
  template<typename VecType>
  math::RangeType<ElemType>
  RangeDistance(const VecType& point,
                typename boost::enable_if<IsVector<VecType> >::type* = 0) const
  {
    return bound.RangeDistance(point);
  }

  //! Return the index of the beginning point of this subset.
  size_t Begin() const { return begin; }
  //! Modify the index of the beginning point of this subset.
  size_t& Begin() { return begin; }

  //! Return the number of points in this subset.
  size_t Count() const { return count; }
  //! Modify the number of points in this subset.
  size_t& Count() { return count; }

  //! Store the center of the bounding region in the given vector.
  void Center(arma::vec& center) const { bound.Center(center); }

 private:
  /**
   * Splits the current node, assigning its left and right children recursively.
   *
   * @param maxLeafSize Maximum number of points held in a leaf.
   * @param splitter Instantiated SplitType object.
   */
  void SplitNode(const size_t maxLeafSize,
                 SplitType<BoundType<MetricType>, MatType>& splitter);

  /**
   * Splits the current node, assigning its left and right children recursively.
   * Also returns a list of the changed indices.
   *
   * @param oldFromNew Vector holding permuted indices.
   * @param maxLeafSize Maximum number of points held in a leaf.
   * @param splitter Instantiated SplitType object.
   */
  void SplitNode(std::vector<size_t>& oldFromNew,
                 const size_t maxLeafSize,
                 SplitType<BoundType<MetricType>, MatType>& splitter);

  /**
   * Update the bound of the current node. This method does not take into
   * account bound-specific properties.
   *
   * @param boundToUpdate The bound to update.
   */
  template<typename BoundType2>
  void UpdateBound(BoundType2& boundToUpdate);

  /**
   * Update the bound of the current node. This method is designed for
   * HollowBallBound only.
   *
   * @param boundToUpdate The bound to update.
   */
  void UpdateBound(bound::HollowBallBound<MetricType>& boundToUpdate);

 protected:
  /**
   * A default constructor.  This is meant to only be used with
   * boost::serialization, which is allowed with the friend declaration below.
   * This does not return a valid tree!  The method must be protected, so that
   * the serialization shim can work with the default constructor.
   */
  BinarySpaceTree();

  //! Friend access is given for the default constructor.
  friend class boost::serialization::access;

 public:
  /**
   * Serialize the tree.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int version);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "binary_space_tree_impl.hpp"

// Include everything else, if necessary.
#include "../binary_space_tree.hpp"

#endif
