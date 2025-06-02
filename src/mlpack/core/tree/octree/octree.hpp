/**
 * @file core/tree/octree/octree.hpp
 * @author Ryan Curtin
 *
 * Definition of generalized octree (Octree).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_OCTREE_OCTREE_HPP
#define MLPACK_CORE_TREE_OCTREE_OCTREE_HPP

#include <mlpack/prereqs.hpp>
#include "../hrectbound.hpp"
#include "../statistic.hpp"

namespace mlpack {

template<typename DistanceType = EuclideanDistance,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat>
class Octree
{
 public:
  //! So other classes can use TreeType::Mat.
  using Mat = MatType;
  //! The type of element held in MatType.
  using ElemType = typename MatType::elem_type;

  //! A single-tree traverser; see single_tree_traverser.hpp.
  template<typename RuleType>
  class SingleTreeTraverser;

  //! A dual-tree traverser; see dual_tree_traverser.hpp.
  template<typename RuleType>
  class DualTreeTraverser;

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
  HRectBound<DistanceType, ElemType> bound;
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
  //! An instantiated distance metric.
  DistanceType distance;

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
         const arma::Col<ElemType>& center,
         const ElemType width,
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
         const arma::Col<ElemType>& center,
         const ElemType width,
         const size_t maxLeafSize = 20);

  /**
   * A default constructor.  This is meant to only be used with cereal.  This
   * returns an empty tree with no points!  It is not very useful.
   */
  Octree();

  /**
   * Copy the given tree.  Be careful!  This may use a lot of memory.
   *
   * @param other Tree to copy from.
   */
  Octree(const Octree& other);

  /**
   * Move the given tree.  The tree passed as a parameter will be emptied and
   * will not be usable after this call.
   *
   * @param other Tree to move.
   */
  Octree(Octree&& other);

  /**
   * Copy the given Octree.
   *
   * @param other The tree to be copied.
   */
  Octree& operator=(const Octree& other);

  /**
   * Take ownership of the given Octree.
   *
   * @param other The tree to take ownership of.
   */
  Octree& operator=(Octree&& other);

  /**
   * Initialize the tree from a cereal archive.
   *
   * @param ar Archive to load tree from.  Must be an iarchive, not an oarchive.
   */
  template<typename Archive>
  Octree(
      Archive& ar,
      const typename std::enable_if_t<cereal::is_loading<Archive>()>* = 0);

  /**
   * Destroy the tree.
   */
  ~Octree();

  //! Return the dataset used by this node.
  const MatType& Dataset() const { return *dataset; }

  //! Get the pointer to the parent.
  Octree* Parent() const { return parent; }
  //! Modify the pointer to the parent (be careful!).
  Octree*& Parent() { return parent; }

  //! Return the bound object for this node.
  const HRectBound<DistanceType, ElemType>& Bound() const { return bound; }
  //! Modify the bound object for this node.
  HRectBound<DistanceType, ElemType>& Bound() { return bound; }

  //! Return the statistic object for this node.
  const StatisticType& Stat() const { return stat; }
  //! Modify the statistic object for this node.
  StatisticType& Stat() { return stat; }

  //! Return the number of children in this node.
  size_t NumChildren() const;

  //! Return the distance metric that this tree uses.
  [[deprecated("Will be removed in mlpack 5.0.0; use Distance()")]]
  DistanceType Metric() const { return distance; }

  //! Return the distance metric that this tree uses.
  DistanceType Distance() const { return distance; }

  /**
   * Return the index of the nearest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetNearestChild(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;

  /**
   * Return the index of the furthest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetFurthestChild(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;

  /**
   * Return whether or not the node is a leaf.
   */
  bool IsLeaf() const { return NumChildren() == 0; }

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
  ElemType MinDistance(const Octree& other) const;
  //! Return the maximum distance to another node.
  ElemType MaxDistance(const Octree& other) const;
  //! Return the minimum and maximum distance to another node.
  RangeType<ElemType> RangeDistance(const Octree& other) const;

  //! Return the minimum distance to the given point.
  template<typename VecType>
  ElemType MinDistance(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
  //! Return the maximum distance to the given point.
  template<typename VecType>
  ElemType MaxDistance(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
  //! Return the minimum and maximum distance to another node.
  template<typename VecType>
  RangeType<ElemType> RangeDistance(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;

  //! Store the center of the bounding region in the given vector.
  template<typename VecType>
  void Center(VecType& center) const { bound.Center(center); }

  //! Serialize the tree.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //! Friend access is given for the default constructor.
  friend class cereal::access;

 private:
  /**
   * Split the node, using the given center and the given maximum width of this
   * node.
   *
   * @param center Center of the node.
   * @param width Width of the current node.
   * @param maxLeafSize Maximum number of points allowed in a leaf.
   */
  void SplitNode(const arma::Col<ElemType>& center,
                 const ElemType width,
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
  void SplitNode(const arma::Col<ElemType>& center,
                 const ElemType width,
                 std::vector<size_t>& oldFromNew,
                 const size_t maxLeafSize);

  /**
   * This is used for sorting points while splitting.
   */
  struct SplitType
  {
    struct SplitInfo
    {
      //! Create the SplitInfo object.
      SplitInfo(const size_t d, const arma::Col<ElemType>& c) :
          d(d), center(c) {}

      //! The dimension we are splitting on.
      size_t d;
      //! The center of the node.
      const arma::Col<ElemType>& center;
    };

    template<typename VecType>
    static bool AssignToLeftNode(const VecType& point, const SplitInfo& s)
    {
      return point[s.d] < s.center[s.d];
    }
  };
};

} // namespace mlpack

// Include implementation.
#include "octree_impl.hpp"

#endif
