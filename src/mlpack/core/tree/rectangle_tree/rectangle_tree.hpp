/**
 * @file rectangle_tree.hpp
 * @author Andrew Wells
 *
 * Definition of generalized rectangle type trees (r_tree, r_star_tree, x_tree,
 * and hilbert_r_tree).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_HPP

#include <mlpack/core.hpp>

#include "../hrectbound.hpp"
#include "../statistic.hpp"
#include "r_tree_split.hpp"
#include "r_tree_descent_heuristic.hpp"
#include "no_auxiliary_information.hpp"

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A rectangle type tree tree, such as an R-tree or X-tree.  Once the
 * bound and type of dataset is defined, the tree will construct itself.  Call
 * the constructor with the dataset to build the tree on, and the entire tree
 * will be built.
 *
 * This tree does allow growth, so you can add and delete nodes from it.
 *
 * @tparam MetricType This *must* be EuclideanDistance, but the template
 *     parameter is required to satisfy the TreeType API.
 * @tparam StatisticType Extra data contained in the node.  See statistic.hpp
 *     for the necessary skeleton interface.
 * @tparam MatType The dataset class.
 * @tparam SplitType The type of split to use when inserting points.
 * @tparam DescentType The heuristic to use when descending the tree to insert
 *    points.
 * @tparam AuxiliaryInformationType An auxiliary information contained
 *    in the node. This information depends on the type of the RectangleTree.
 */

template<typename MetricType = metric::EuclideanDistance,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat,
         typename SplitType = RTreeSplit,
         typename DescentType = RTreeDescentHeuristic,
         template<typename> class AuxiliaryInformationType = NoAuxiliaryInformation>
class RectangleTree
{
  // The metric *must* be the euclidean distance.
  static_assert(boost::is_same<MetricType, metric::EuclideanDistance>::value,
      "RectangleTree: MetricType must be metric::EuclideanDistance.");

 public:
  //! So other classes can use TreeType::Mat.
  typedef MatType Mat;
  //! The element type held by the matrix type.
  typedef typename MatType::elem_type ElemType;
  //! The auxiliary information type held by the tree.
  typedef AuxiliaryInformationType<RectangleTree> AuxiliaryInformation;
 private:
  //! The max number of child nodes a non-leaf node can have.
  size_t maxNumChildren;
  //! The minimum number of child nodes a non-leaf node can have.
  size_t minNumChildren;
  //! The number of child nodes actually in use (0 if this is a leaf node).
  size_t numChildren;
  //! The child nodes (Starting at 0 and ending at (numChildren-1) ).
  std::vector<RectangleTree*> children;
  //! The parent node (NULL if this is the root of the tree).
  RectangleTree* parent;
  //! The index of the first point in the dataset contained in this node (and
  //! its children).  THIS IS ALWAYS 0 AT THE MOMENT.  IT EXISTS MERELY IN CASE
  //! I THINK OF A WAY TO CHANGE THAT.  IN OTHER WORDS, IT WILL PROBABLY BE
  //! REMOVED.
  size_t begin;
  //! The number of points in the dataset contained in this node (and its
  //! children).
  size_t count;
  //! The number of descendants of this node.
  size_t numDescendants;
  //! The max leaf size.
  size_t maxLeafSize;
  //! The minimum leaf size.
  size_t minLeafSize;
  //! The bound object for this node.
  bound::HRectBound<metric::EuclideanDistance, ElemType> bound;
  //! Any extra data contained in the node.
  StatisticType stat;
  //! The distance from the centroid of this node to the centroid of the parent.
  ElemType parentDistance;
  //! The dataset.
  const MatType* dataset;
  //! Whether or not we are responsible for deleting the dataset.  This is
  //! probably not aligned well...
  bool ownsDataset;
  //! The mapping to the dataset
  std::vector<size_t> points;
  //! A tree-specific information
  AuxiliaryInformationType<RectangleTree> auxiliaryInfo;

 public:
  //! A single traverser for rectangle type trees.  See
  //! single_tree_traverser.hpp for implementation.
  template<typename RuleType>
  class SingleTreeTraverser;
  //! A dual tree traverser for rectangle type trees.
  template<typename RuleType>
  class DualTreeTraverser;

  /**
   * Construct this as the root node of a rectangle type tree using the given
   * dataset.  This will modify the ordering of the points in the dataset!
   *
   * @param data Dataset from which to create the tree.  This will be modified!
   * @param maxLeafSize Maximum size of each leaf in the tree.
   * @param minLeafSize Minimum size of each leaf in the tree.
   * @param maxNumChildren The maximum number of child nodes a non-leaf node may
   *      have.
   * @param minNumChildren The minimum number of child nodes a non-leaf node may
   *      have.
   * @param firstDataIndex The index of the first data point.  UNUSED UNLESS WE
   *      ADD SUPPORT FOR HAVING A "CENTERAL" DATA MATRIX.
   */
  RectangleTree(const MatType& data,
                const size_t maxLeafSize = 20,
                const size_t minLeafSize = 8,
                const size_t maxNumChildren = 5,
                const size_t minNumChildren = 2,
                const size_t firstDataIndex = 0);

  /**
   * Construct this as the root node of a rectangle tree type using the given
   * dataset, and taking ownership of the given dataset.
   *
   * @param data Dataset from which to create the tree.
   * @param maxLeafSize Maximum size of each leaf in the tree.
   * @param minLeafSize Minimum size of each leaf in the tree.
   * @param maxNumChildren The maximum number of child nodes a non-leaf node may
   *      have.
   * @param minNumChildren The minimum number of child nodes a non-leaf node may
   *      have.
   * @param firstDataIndex The index of the first data point.  UNUSED UNLESS WE
   *      ADD SUPPORT FOR HAVING A "CENTERAL" DATA MATRIX.
   */
  RectangleTree(MatType&& data,
                const size_t maxLeafSize = 20,
                const size_t minLeafSize = 8,
                const size_t maxNumChildren = 5,
                const size_t minNumChildren = 2,
                const size_t firstDataIndex = 0);

  /**
   * Construct this as an empty node with the specified parent.  Copying the
   * parameters (maxLeafSize, minLeafSize, maxNumChildren, minNumChildren,
   * firstDataIndex) from the parent.
   *
   * @param parentNode The parent of the node that is being constructed.
   * @param numMaxChildren The max number of child nodes (used in x-trees).
   */
  explicit RectangleTree(RectangleTree* parentNode,
                         const size_t numMaxChildren = 0);

  /**
   * Create a rectangle tree by copying the other tree.  Be careful!  This can
   * take a long time and use a lot of memory.
   *
   * @param other The tree to be copied.
   * @param deepCopy If false, the children are not recursively copied.
   */
  RectangleTree(const RectangleTree& other,
                const bool deepCopy = true,
                RectangleTree* newParent = NULL);

  /**
   * Create a rectangle tree by moving the other tree.
   *
   * @param other The tree to be copied.
   */
  RectangleTree(RectangleTree&& other);

  /**
   * Construct the tree from a boost::serialization archive.
   */
  template<typename Archive>
  RectangleTree(
      Archive& ar,
      const typename boost::enable_if<typename Archive::is_loading>::type* = 0);

  /**
   * Deletes this node, deallocating the memory for the children and calling
   * their destructors in turn.  This will invalidate any younters or references
   * to any nodes which are children of this one.
   */
  ~RectangleTree();

  /**
   * Delete this node of the tree, but leave the stuff contained in it intact.
   * This is used when splitting a node, where the data in this tree is moved to
   * two other trees.
   */
  void SoftDelete();

  /**
   * Nullify the auxiliary information. Used for memory management.
   * Be cafeful.
   */
  void NullifyData();

  /**
   * Inserts a point into the tree.
   *
   * @param point The index of a point in the dataset.
   */
  void InsertPoint(const size_t point);

  /**
   * Inserts a point into the tree, tracking which levels have been inserted
   * into.
   *
   * @param point The index of a point in the dataset.
   * @param relevels The levels that have been reinserted to on this top level
   *      insertion.
   */
  void InsertPoint(const size_t point, std::vector<bool>& relevels);

  /**
   * Inserts a node into the tree, tracking which levels have been inserted
   * into.  The node will be inserted so that the tree remains valid.
   *
   * @param node The node to be inserted.
   * @param level The depth that should match the node where this node is
   *      finally inserted.  This should be the number returned by calling
   *      TreeDepth() from the node that originally contained "node".
   * @param relevels The levels that have been reinserted to on this top level
   *      insertion.
   */
  void InsertNode(RectangleTree* node,
                  const size_t level,
                  std::vector<bool>& relevels);

  /**
   * Deletes a point from the treeand, updates the bounding rectangle.
   * However, the point will be kept in the centeral dataset. (The
   * user may remove it from there if he wants, but he must not change the
   * indices of the other points.) Returns true if the point is successfully
   * removed and false if it is not.  (ie. the point is not in the tree)
   */
  bool DeletePoint(const size_t point);

  /**
   * Deletes a point from the tree, updates the bounding rectangle,
   * tracking levels. However, the point will be kept in the centeral dataset.
   * (The user may remove it from there if he wants, but he
   * must not change the indices of the other points.) Returns true if the point
   * is successfully removed and false if it is not.  (ie. the point is not in
   * the tree)
   */
  bool DeletePoint(const size_t point, std::vector<bool>& relevels);

  /**
   * Removes a node from the tree.  You are responsible for deleting it if you
   * wish to do so.
   */
  bool RemoveNode(const RectangleTree* node, std::vector<bool>& relevels);

  /**
   * Find a node in this tree by its begin and count (const).
   *
   * Every node is uniquely identified by these two numbers.
   * This is useful for communicating position over the network,
   * when pointers would be invalid.
   *
   * @param begin The begin() of the node to find.
   * @param count The count() of the node to find.
   * @return The found node, or NULL if not found.
   */
  const RectangleTree* FindByBeginCount(size_t begin, size_t count) const;

  /**
   * Find a node in this tree by its begin and count.
   *
   * Every node is uniquely identified by these two numbers.
   * This is useful for communicating position over the network,
   * when pointers would be invalid.
   *
   * @param begin The begin() of the node to find.
   * @param count The count() of the node to find.
   * @return The found node, or NULL if not found.
   */
  RectangleTree* FindByBeginCount(size_t begin, size_t count);

  //! Return the bound object for this node.
  const bound::HRectBound<MetricType>& Bound() const { return bound; }
  //! Modify the bound object for this node.
  bound::HRectBound<MetricType>& Bound() { return bound; }

  //! Return the statistic object for this node.
  const StatisticType& Stat() const { return stat; }
  //! Modify the statistic object for this node.
  StatisticType& Stat() { return stat; }

  //! Return the auxiliary information object of this node.
  const AuxiliaryInformationType<RectangleTree> &AuxiliaryInfo() const
  { return auxiliaryInfo; }
  //! Modify the split object of this node.
  AuxiliaryInformationType<RectangleTree>& AuxiliaryInfo()
  { return auxiliaryInfo; }

  //! Return whether or not this node is a leaf (true if it has no children).
  bool IsLeaf() const;

  //! Return the maximum leaf size.
  size_t MaxLeafSize() const { return maxLeafSize; }
  //! Modify the maximum leaf size.
  size_t& MaxLeafSize() { return maxLeafSize; }

  //! Return the minimum leaf size.
  size_t MinLeafSize() const { return minLeafSize; }
  //! Modify the minimum leaf size.
  size_t& MinLeafSize() { return minLeafSize; }

  //! Return the maximum number of children (in a non-leaf node).
  size_t MaxNumChildren() const { return maxNumChildren; }
  //! Modify the maximum number of children (in a non-leaf node).
  size_t& MaxNumChildren() { return maxNumChildren; }

  //! Return the minimum number of children (in a non-leaf node).
  size_t MinNumChildren() const { return minNumChildren; }
  //! Modify the minimum number of children (in a non-leaf node).
  size_t& MinNumChildren() { return minNumChildren; }

  //! Gets the parent of this node.
  RectangleTree* Parent() const { return parent; }
  //! Modify the parent of this node.
  RectangleTree*& Parent() { return parent; }

  //! Get the dataset which the tree is built on.
  const MatType& Dataset() const { return *dataset; }
  //! Modify the dataset which the tree is built on.  Be careful!
  MatType& Dataset() { return const_cast<MatType&>(*dataset); }

  //! Get the metric which the tree uses.
  MetricType Metric() const { return MetricType(); }

  //! Get the centroid of the node and store it in the given vector.
  void Center(arma::vec& center) { bound.Center(center); }

  //! Return the number of child nodes.  (One level beneath this one only.)
  size_t NumChildren() const { return numChildren; }
  //! Modify the number of child nodes.  Be careful.
  size_t& NumChildren() { return numChildren; }

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
  size_t GetNearestChild(const RectangleTree& queryNode);

  /**
   * Return the index of the furthest child node to the given query node.  If it
   * can't decide, it will return NumChildren() (invalid index).
   */
  size_t GetFurthestChild(const RectangleTree& queryNode);

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

  //! Return the minimum distance from the center to any edge of the bound.
  //! Currently, this returns 0, which doesn't break algorithms, but it isn't
  //! necessarily correct, either.
  ElemType MinimumBoundDistance() const { return bound.MinWidth() / 2.0; }

  //! Return the distance from the center of this node to the center of the
  //! parent node.
  ElemType ParentDistance() const { return parentDistance; }
  //! Modify the distance from the center of this node to the center of the
  //! parent node.
  ElemType& ParentDistance() { return parentDistance; }

  /**
   * Get the specified child.
   *
   * @param child Index of child to return.
   */
  inline RectangleTree& Child(const size_t child) const
  {
    return *children[child];
  }

  /**
   * Modify the specified child.
   *
   * @param child Index of child to return.
   */
  inline RectangleTree& Child(const size_t child)
  {
    return *children[child];
  }

  //! Return the number of points in this node (returns 0 if this node is not a
  //! leaf).
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
  size_t Point(const size_t index) const { return points[index]; }

  //! Modify the index of a particular point in this node.  Be very careful when
  //! you do this!  You may make the tree invalid.
  size_t& Point(const size_t index) { return points[index]; }

  //! Return the minimum distance to another node.
  ElemType MinDistance(const RectangleTree& other) const
  {
    return bound.MinDistance(other.Bound());
  }

  //! Return the maximum distance to another node.
  ElemType MaxDistance(const RectangleTree& other) const
  {
    return bound.MaxDistance(other.Bound());
  }

  //! Return the minimum and maximum distance to another node.
  math::RangeType<ElemType> RangeDistance(const RectangleTree& other) const
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
  math::RangeType<ElemType> RangeDistance(
      const VecType& point,
      typename boost::enable_if<IsVector<VecType> >::type* = 0) const
  {
    return bound.RangeDistance(point);
  }

  /**
   * Obtains the number of nodes in the tree, starting with this.
   */
  size_t TreeSize() const;

  /**
   * Obtains the number of levels below this node in the tree, starting with
   * this.
   */
  size_t TreeDepth() const;

  //! Return the index of the beginning point of this subset.
  size_t Begin() const { return begin; }
  //! Modify the index of the beginning point of this subset.
  size_t& Begin() { return begin; }

  //! Return the number of points in this subset.
  size_t Count() const { return count; }
  //! Modify the number of points in this subset.
  size_t& Count() { return count; }

 private:
  /**
   * Splits the current node, recursing up the tree.
   *
   * @param relevels Vector to track which levels have been inserted to.
   */
  void SplitNode(std::vector<bool>& relevels);

 protected:
  /**
   * A default constructor.  This is meant to only be used with
   * boost::serialization, which is allowed with the friend declaration below.
   * This does not return a valid tree!  This method must be protected, so that
   * the serialization shim can work with the default constructor.
   */
  RectangleTree();

  //! Friend access is given for the default constructor.
  friend class boost::serialization::access;

  //! Give friend access for DescentType.
  friend DescentType;

  //! Give friend access for SplitType.
  friend SplitType;

  //! Give friend access for AuxiliaryInformationType.
  friend AuxiliaryInformation;

 public:
  /**
   * Condense the bounding rectangles for this node based on the removal of the
   * point specified by the arma::vec&.  This recurses up the tree.  If a node
   * goes below the minimum fill, this function will fix the tree.
   *
   * @param point The arma::vec& of the point that was removed to require this
   *      condesation of the tree.
   * @param usePoint True if we use the optimized version of the algorithm that
   *      is possible when we now what point was deleted.  False otherwise (eg.
   *      if we deleted a node instead of a point).
   */
  void CondenseTree(const arma::vec& point,
                    std::vector<bool>& relevels,
                    const bool usePoint);

  /**
   * Shrink the bound object of this node for the removal of a point.
   *
   * @param point The arma::vec& of the point that was removed to require this
   *      shrinking.
   * @return true if the bound needed to be changed, false if it did not.
   */
  bool ShrinkBoundForPoint(const arma::vec& point);

  /**
   * Shrink the bound object of this node for the removal of a child node.
   *
   * @param bound The HRectBound<>& of the bound that was removed to reqire this
   *      shrinking.
   * @return true if the bound needed to be changed, false if it did not.
   */
  bool ShrinkBoundForBound(const bound::HRectBound<MetricType>& changedBound);

  /**
   * Make an exact copy of this node, pointers and everything.
   */
  RectangleTree* ExactClone();

  /**
   * Serialize the tree.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "rectangle_tree_impl.hpp"

#endif
