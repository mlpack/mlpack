/**
 * @file octree.hpp
 * @author Ryan Curtin
 *
 * Definition of generalized octree (Octree).
 */
#ifndef MLPACK_CORE_TREE_OCTREE_OCTREE_HPP
#define MLPACK_CORE_TREE_OCTREE_OCTREE_HPP

#include <mlpack/core.hpp>
#include "../hrectbound.hpp"
#include "../statistic.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType = metric::EuclideanDistance,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat>
class Octree
{
 public:
  //! So other classes can use TreeType::Mat.
  typedef MatType Mat;
  //! The type of element held in MatType.
  typedef typename MatType::elem_type ElemType;

 private:
  //! The children held by this node.
  std::vector<Octree*> children;

  //! The index of the first point in the dataset contained in this node (and
  //! its children).
  size_t begin;
  //! The number of points of the dataset contained in this node (and its
  //! children).
  size_t count;
  //! The minimum bounding rectangle of the points held in the node (and its
  //! children).
  bound::HRectBound<MetricType> bound;
  //! The dataset.
  MatType* dataset;
  //! The parent (NULL if this node is the root).
  Octree* parent;
  //! The statistic.
  StatisticType stat;
  //! The distance from the center of this node to the center of the parent.
  ElemType parentDistance;
  //! The distance to the furthest descendant, cached to speed things up.
  ElemType furthestDescendantDistance;
  //! An instantiated metric.
  MetricType metric;

 public:
  /**
   * Construct this as the root node of an octree on the given dataset.  This
   * copies the dataset.  If you don't want to copy the input dataset, consider
   * using the constructor that takes an rvalue reference and use std::move().
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param maxLeafSize Maximum number of points in a leaf node.
   */
  Octree(const MatType& data, const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of an octree on the given dataset.  This
   * copies the dataset and modifies its ordering; a mapping of the old point
   * indices to the new point indices is filled.  If you don't want the matrix
   * to be copied, consider using the constructor that takes an rvalue reference
   * and use std::move().
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param oldFromNew Vector which will be filled with the old positions for
   *      each new point.
   * @param maxLeafSize Maximum number of points in a leaf node.
   */
  Octree(const MatType& data,
         std::vector<size_t>& oldFromNew,
         const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of an octree on the given dataset.  This
   * copies the dataset and modifies its ordering; a mapping of the old point
   * indices to the new point indices is filled, and a mapping of the new point
   * indices to the old point indices is filled.  If you don't want the matrix
   * to be copied, consider using the constructor that takes an rvalue reference
   * and use std::move().
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param oldFromNew Vector which will be filled with the old positions for
   *      each new point.
   * @param newFromOld Vector which will be filled with the new positions for
   *      each old point.
   * @param maxLeafSize Maximum number of points in a leaf node.
   */
  Octree(const MatType& data,
         std::vector<size_t>& oldFromNew,
         std::vector<size_t>& newFromOld,
         const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of an octree on the given dataset.  This
   * will take ownership of the dataset; if you don't want this, consider using
   * the constructor that takes a const reference to the dataset.
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param maxLeafSize Maximum number of points in a leaf node.
   */
  Octree(MatType&& data, const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of an octree on the given dataset. This
   * will take ownership of the dataset; if you don't want this, consider using
   * the constructor that takes a const reference to the dataset.  This modifies
   * the ordering of the dataset; a mapping of the old point indices to the new
   * point indices is filled.
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param oldFromNew Vector which will be filled with the old positions for
   *      each new point.
   * @param maxLeafSize Maximum number of points in a leaf node.
   */
  Octree(MatType&& data,
         std::vector<size_t>& oldFromNew,
         const size_t maxLeafSize = 20);

  /**
   * Construct this as the root node of an octree on the given dataset.  This
   * will take ownership of the dataset; if you don't want this, consider using
   * the constructor that takes a const reference to the dataset.  This modifies
   * the ordering of the dataset; a mapping of the old point indices to the new
   * point indices is filled, and a mapping of the new point indices to the old
   * point indices is filled.
   *
   * @param data Dataset to create tree from.  This will be copied!
   * @param oldFromNew Vector which will be filled with the old positions for
   *      each new point.
   * @param newFromOld Vector which will be filled with the new positions for
   *      each old point.
   * @param maxLeafSize Maximum number of points in a leaf node.
   */
  Octree(MatType&& data,
         std::vector<size_t>& oldFromNew,
         std::vector<size_t>& newFromOld,
         const size_t maxLeafSize = 20);

  /**
   * Construct this node as a child of the given parent, starting at column
   * begin and using count points.  The ordering of that subset of points in the
   * parent's data matrix will be modified!  This is used for recursive
   * tree-building by the other constructors that don't specify point indices.
   *
   * @param parent Parent of this node.  Its dataset will be modified!
   * @param begin Index of point to start tree construction with.
   * @param count Number of points to use to construct tree.
   * @param center Center of the node (for splitting).
   * @param width Width of the node in each dimension.
   * @param maxLeafSize Maximum number of points in a leaf node.
   */
  Octree(Octree* parent,
         const size_t begin,
         const size_t count,
         const arma::vec& center,
         const double width,
         const size_t maxLeafSize = 20);

  /**
   * Construct this node as a child of the given parent, starting at column
   * begin and using count points.  The ordering of that subset of points in the
   * parent's data matrix will be modified!  This is used for recursive
   * tree-building by the other constructors that don't specify point indices.
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
   *      each new point.
   * @param center Center of the node (for splitting).
   * @param width Width of the node in each dimension.
   * @param maxLeafSize Maximum number of points in a leaf node.
   */
  Octree(Octree* parent,
         const size_t begin,
         const size_t count,
         std::vector<size_t>& oldFromNew,
         const arma::vec& center,
         const double width,
         const size_t maxLeafSize = 20);

  /**
   * Destroy the tree.
   */
  ~Octree();

  //! Return the dataset used by this node.
  const MatType& Dataset() const { return *dataset; }

  //! Return the bound object for this node.
  const bound::HRectBound<MetricType>& Bound() const { return bound; }
  //! Modify the bound object for this node.
  bound::HRectBound<MetricType>& Bound() { return bound; }

  //! Return the statistic object for this node.
  const StatisticType& Stat() const { return stat; }
  //! Modify the statistic object for this node.
  StatisticType& Stat() { return stat; }

  //! Return the number of children in this node.
  size_t NumChildren() const;

  //! Return the metric that this tree uses.
  MetricType Metric() const { return MetricType(); }

  /**
   * Return the index of the nearest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetNearestChild(
      const VecType& point,
      typename boost::enable_if<IsVector<VecType>>::type* = 0) const;

  /**
   * Return the index of the furthest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetFurthestChild(
      const VecType& point,
      typename boost::enable_if<IsVector<VecType> >::type* = 0) const;

  /**
   * Return the index of the nearest child node to the given query node.  If it
   * can't decide, it will return NumChildren() (invalid index).
   */
  size_t GetNearestChild(const Octree& queryNode) const;

  /**
   * Return the index of the furthest child node to the given query node.  If it
   * can't decide, it will return NumChildren() (invalid index).
   */
  size_t GetFurthestChild(const Octree& queryNode) const;

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
   * Return the specified child.  If the index is out of bounds, unspecified
   * behavior will occur.
   */
  const Octree& Child(const size_t child) const { return *children[child]; }

  /**
   * Return the specified child.  If the index is out of bounds, unspecified
   * behavior will occur.
   */
  Octree& Child(const size_t child) { return *children[child]; }

  /**
   * Return the pointer to the given child.  This allows the child itself to be
   * modified.
   */
  Octree*& ChildPtr(const size_t child) { return children[child]; }

  //! Return the number of points in this node (0 if not a leaf).
  size_t NumPoints() const;

  //! Return the number of descendants of this node.
  size_t NumDescendants() const;

  /**
   * Return the index (with reference to the dataset) of a particular
   * descendant.
   */
  size_t Descendant(const size_t index) const;

  /**
   * Return the index (with reference to the dataset) of a particular point in
   * this node.  If the given index is invalid (i.e. if it is greater than
   * NumPoints()), the indices returned will be invalid.
   */
  size_t Point(const size_t index) const;

  //! Return the minimum distance to another node.
  ElemType MinDistance(const Octree* other) const;
  //! Return the maximum distance to another node.
  ElemType MaxDistance(const Octree* other) const;
  //! Return the minimum and maximum distance to another node.
  math::RangeType<ElemType> RangeDistance(const Octree* other) const;

  //! Return the minimum distance to the given point.
  template<typename VecType>
  ElemType MinDistance(
      const VecType& point,
      typename boost::enable_if<IsVector<VecType>>::type* = 0) const;
  //! Return the maximum distance to the given point.
  template<typename VecType>
  ElemType MaxDistance(
      const VecType& point,
      typename boost::enable_if<IsVector<VecType>>::type* = 0) const;
  //! Return the minimum and maximum distance to another node.
  template<typename VecType>
  math::RangeType<ElemType> RangeDistance(
      const VecType& point,
      typename boost::enable_if<IsVector<VecType>>::type* = 0) const;

  //! Store the center of the bounding region in the given vector.
  void Center(arma::vec& center) const { bound.Center(center); }

 private:
  /**
   * Split the node, using the given center and the given maximum width of this
   * node.
   *
   * @param center Center of the node.
   * @param width Width of the current node.
   * @param maxLeafSize Maximum number of points allowed in a leaf.
   */
  void SplitNode(const arma::vec& center,
                 const double width,
                 const size_t maxLeafSize);

  /**
   * Split the node, using the given center and the given maximum width of this
   * node, and fill the mappings vector.
   *
   * @param center Center of the node.
   * @param width Width of the current node.
   * @param oldFromNew Mappings from old to new.
   * @param maxLeafSize Maximum number of points allowed in a leaf.
   */
  void SplitNode(const arma::vec& center,
                 const double width,
                 std::vector<size_t>& oldFromNew,
                 const size_t maxLeafSize);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "octree_impl.hpp"

#endif
