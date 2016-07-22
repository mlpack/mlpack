/**
 * @file vantage_point_tree.hpp
 *
 * Definition of the vantage point tree.
 */

#ifndef MLPACK_CORE_TREE_VANTAGE_POINT_TREE_VANTAGE_POINT_TREE_HPP
#define MLPACK_CORE_TREE_VANTAGE_POINT_TREE_VANTAGE_POINT_TREE_HPP

#include "vantage_point_split.hpp"

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * The vantage point tree is a variant of a binary space tree. The difference
 * from BinarySpaceTree is a presence of points in intermediate nodes.
 * If an intermediate node holds a point, this point is the centroid of the
 * bound.
 *
 * This particular tree does not allow growth, so you cannot add or delete nodes
 * from it.  If you need to add or delete a node, the better procedure is to
 * rebuild the tree entirely.
 *
 * This tree does take one runtime parameter in the constructor, which is the
 * max leaf size to be used.

  * @tparam MetricType The metric used for tree-building.  The BoundType may
 *     place restrictions on the metrics that can be used.
 * @tparam StatisticType Extra data contained in the node.  See statistic.hpp
 *     for the necessary skeleton interface.
 * @tparam MatType The dataset class.
 * @tparam BoundType The bound used for each node.  Currently only
 *    HollowBallBound is supported.
 * @tparam SplitType The class that partitions the dataset/points at a
 *     particular node into two parts. Its definition decides the way this split
 *     is done.
 */
template<typename MetricType,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat,
         template<typename BoundMetricType, typename...> class BoundType =
            bound::HollowBallBound,
         template<typename SplitBoundType, typename SplitMatType, size_t...>
            class SplitType = VantagePointSplit>
class VantagePointTree
{
 public:
  //! So other classes can use TreeType::Mat.
  typedef MatType Mat;
  //! The type of element held in MatType.
  typedef typename MatType::elem_type ElemType;

 private:
  //! The left child node.
  VantagePointTree* left;
  //! The right child node.
  VantagePointTree* right;
  //! The parent node (NULL if this is the root of the tree).
  VantagePointTree* parent;
  //! The index of the first point in the dataset contained in this node.
  size_t begin;
  //! The number of points of the dataset contained in this node.
  size_t count;
  //! The bound object for this node.
  BoundType<MetricType, ElemType> bound;
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
  //! Indicates that the first point of the node is the centroid of its bound.
  bool firstPointIsCentroid;

 public:
  //! A single-tree traverser for the vantage point tree; see
  //! single_tree_traverser.hpp for implementation.
  template<typename RuleType>
  class SingleTreeTraverser;

  //! A dual-tree traverser for the vantage point tree;
  //! see dual_tree_traverser.hpp.
  template<typename RuleType>
  class DualTreeTraverser;

  /**
   * Construct this as the root node of a vantage point tree using the given
   * dataset.  This will copy the input matrix; if you don't want this, consider
   * using the constructor that takes an rvalue reference and use std::move().
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param maxLeafSize Size of each leaf in the tree.
   */
  VantagePointTree(const MatType& data, const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a vantage point tree using the given
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
  VantagePointTree(const MatType& data,
                   std::vector<size_t>& oldFromNew,
                   const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a vantage point tree using the given
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
  VantagePointTree(const MatType& data,
                   std::vector<size_t>& oldFromNew,
                   std::vector<size_t>& newFromOld,
                   const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a vantage point tree using the given
   * dataset.  This will take ownership of the data matrix; if you don't want
   * this, consider using the constructor that takes a const reference to a
   * dataset.
   *
   * @param data Dataset to create tree from.
   * @param maxLeafSize Size of each leaf in the tree.
   */
  VantagePointTree(MatType&& data,
                   const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a vantage point tree using the given
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
  VantagePointTree(MatType&& data,
                   std::vector<size_t>& oldFromNew,
                   const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of a vantage point tree using the given
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
  VantagePointTree(MatType&& data,
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
   * @param maxLeafSize Size of each leaf in the tree.
   * @param firstPointIsCentroid Indicates that the first point of the node is
   *    the centroid of its bound.
   */
  VantagePointTree(VantagePointTree* parent,
                   const size_t begin,
                   const size_t count,
                   SplitType<BoundType<MetricType>, MatType>& splitter,
                   const size_t maxLeafSize = 20,
                   bool firstPointIsCentroid = false);

  /**
   * Construct this node as a child of the given parent, starting at column
   * begin and using count points.  The ordering of that subset of points in the
   * parent's data matrix will be modified!  This is used for recursive
   * tree-building by the other constructors which don't specify point indices.
   *
   * A mapping of the old point indices to the new point indices is filled, but
   * it is expected that the vector is already allocated with size greater than
   * or equal to (begin_in + count_in), and if that is not true, invalid memory
   * reads (and writes) will occur.
   *
   * @param parent Parent of this node.  Its dataset will be modified!
   * @param begin Index of point to start tree construction with.
   * @param count Number of points to use to construct tree.
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param maxLeafSize Size of each leaf in the tree.
   * @param firstPointIsCentroid Indicates that the first point of the node is
   *    the centroid of its bound.
   */
  VantagePointTree(VantagePointTree* parent,
                   const size_t begin,
                   const size_t count,
                   std::vector<size_t>& oldFromNew,
                   SplitType<BoundType<MetricType>, MatType>& splitter,
                   const size_t maxLeafSize = 20,
                   bool firstPointIsCentroid = false);

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
   * @param firstPointIsCentroid Indicates that the first point of the node is
   *    the centroid of its bound.
   */
  VantagePointTree(VantagePointTree* parent,
                   const size_t begin,
                   const size_t count,
                   std::vector<size_t>& oldFromNew,
                   std::vector<size_t>& newFromOld,
                   SplitType<BoundType<MetricType>, MatType>& splitter,
                   const size_t maxLeafSize = 20,
                   bool firstPointIsCentroid = false);

  /**
   * Create a vantage point tree by copying the other tree.  Be careful!  This
   * can take a long time and use a lot of memory.
   *
   * @param other Tree to be replicated.
   */
  VantagePointTree(const VantagePointTree& other);

  /**
   * Move constructor for a VantagePointTree; possess all the members of the
   * given tree.
   */
  VantagePointTree(VantagePointTree&& other);

  /**
   * Initialize the tree from a boost::serialization archive.
   *
   * @param ar Archive to load tree from.  Must be an iarchive, not an oarchive.
   */
  template<typename Archive>
  VantagePointTree(
      Archive& ar,
      const typename boost::enable_if<typename Archive::is_loading>::type* = 0);

  /**
   * Deletes this node, deallocating the memory for the children and calling
   * their destructors in turn.  This will invalidate any pointers or references
   * to any nodes which are children of this one.
   */
  ~VantagePointTree();

  //! Return the bound object for this node.
  const BoundType<MetricType>& Bound() const { return bound; }
  //! Return the bound object for this node.
  BoundType<MetricType>& Bound() { return bound; }

  //! Return the statistic object for this node.
  const StatisticType& Stat() const { return stat; }
  //! Modify the statistic object for this node.
  StatisticType& Stat() { return stat; }

  //! Return whether or not this node is a leaf (true if it has no children).
  bool IsLeaf() const;

  //! Gets the left child of this node.
  VantagePointTree* Left() const { return left; }
  //! Modify the left child of this node.
  VantagePointTree*& Left() { return left; }

  //! Gets the right child of this node.
  VantagePointTree* Right() const { return right; }
  //! Modify the right child of this node.
  VantagePointTree*& Right() { return right; }

  //! Gets the parent of this node.
  VantagePointTree* Parent() const { return parent; }
  //! Modify the parent of this node.
  VantagePointTree*& Parent() { return parent; }

  //! Get the dataset which the tree is built on.
  const MatType& Dataset() const { return *dataset; }
  //! Modify the dataset which the tree is built on.  Be careful!
  MatType& Dataset() { return *dataset; }

  //! Get the metric that the tree uses.
  MetricType Metric() const { return MetricType(); }

  //! Return the number of children in this node.
  size_t NumChildren() const;

  /**
   * Return the furthest distance to a point held in this node.  If this is not
   * a leaf node, then the distance is 0 because the node holds no points or
   * the only point is the centroid.
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
  VantagePointTree& Child(const size_t child) const;

  VantagePointTree*& ChildPtr(const size_t child)
  { return (child == 0) ? left : right; }

  //! Return the number of points in this node.
  size_t NumPoints() const;

  /**
   * Return the number of descendants of this node.
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
  ElemType MinDistance(const VantagePointTree* other) const
  {
    return bound.MinDistance(other->Bound());
  }

  //! Return the maximum distance to another node.
  ElemType MaxDistance(const VantagePointTree* other) const
  {
    return bound.MaxDistance(other->Bound());
  }

  //! Return the minimum and maximum distance to another node.
  math::RangeType<ElemType> RangeDistance(const VantagePointTree* other) const
  {
    return bound.RangeDistance(other->Bound());
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

  //! Returns false: this tree type does not have self children.
  static bool HasSelfChildren() { return false; }

  //! Store the center of the bounding region in the given vector.
  void Center(arma::vec& center) { bound.Center(center); }

  //! Indicates that the first point of this node is the centroid of its bound.
  bool FirstPointIsCentroid() const { return firstPointIsCentroid; }

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

 protected:
  /**
   * A default constructor.  This is meant to only be used with
   * boost::serialization, which is allowed with the friend declaration below.
   * This does not return a valid tree!  The method must be protected, so that
   * the serialization shim can work with the default constructor.
   */
  VantagePointTree();

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
#include "vantage_point_tree_impl.hpp"

#endif // MLPACK_CORE_TREE_VANTAGE_POINT_TREE_VANTAGE_POINT_TREE_HPP
