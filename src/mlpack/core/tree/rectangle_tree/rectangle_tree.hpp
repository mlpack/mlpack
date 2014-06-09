/**
 * @file rectangle_tree.hpp
 * @author Andrew Wells
 *
 * Definition of generalized rectangle type trees (r_tree, r_star_tree, x_tree, and hilbert_r_tree).
 */
#ifndef __MLPACK_CORE_TREE_RECTINGLE_TREE_RECTANGLE_TREE_HPP
#define __MLPACK_CORE_TREE_RECTINGLE_TREE_RECTANGLE_TREE_HPP

#include <mlpack/core.hpp>

#include "../statistic.hpp"

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A rectangle type tree tree, such as an R-tree or X-tree.  Once the
 * bound and type of dataset is defined, the tree will construct itself.  Call
 * the constructor with the dataset to build the tree on, and the entire tree
 * will be built.
 *
 * This tree does allow growth, so you can add and delete nodes
 * from it.
 *
 * @tparam StatisticType Extra data contained in the node.  See statistic.hpp
 *     for the necessary skeleton interface.
 * @tparam MatType The dataset class.
 * @tparam SplitType The type of split to use when inserting points.
 * @tparam DescentType The heuristic to use when descending the tree to insert points.
 */   

template<typename StatisticType = EmptyStatistic,
	 typename MatType = arma::mat,
	 typename SplitType = EmptySplit,
	 typename DescentType = EmptyDescent>
class RectangleTree
{
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
  //! The max leaf size.
  size_t maxLeafSize;
  //! The minimum leaf size.
  size_t minLeafSize;
  //! The bound object for this node.
  HRectBound bound;
  //! Any extra data contained in the node.
  StatisticType stat;
  //! The distance from the centroid of this node to the centroid of the parent.
  double parentDistance;
  //! The discance to the furthest descendant, cached to speed things up.
  double furthestDescendantDistance;
  //! The dataset.
  MatType& dataset;
  
 public:
  //! So other classes can use TreeType::Mat.
  typedef MatType Mat;

  //! A traverser for rectangle type trees.  See
  //! rectangle_tree_traverser.hpp for implementation.
  template<typename RuleType>
  class RectangleTreeTraverser;

  /**
   * Construct this as the root node of a rectangle type tree using the given
   * dataset.  This will modify the ordering of the points in the dataset!
   *
   * @param data Dataset from which to create the tree.  This will be modified!
   * @param maxLeafSize Maximum size of each leaf in the tree;
   * @param maxNumChildren The maximum number of child nodes a non-leaf node may have.
   */
  RectangleTree(MatType& data, const size_t maxLeafSize = 20, const size_t maxNumChildren = 4);

  //TODO implement the oldFromNew stuff if applicable.

  /**
   * Deletes this node, deallocating the memory for the children and calling
   * their destructors in turn.  This will invalidate any younters or references
   * to any nodes which are children of this one.
   */
  ~RectangleTree();
  
  /**
   * Delete this node of the tree, but leave the stuff contained in it intact.
   * This is used when splitting a node, where the data in this tree is moved to two
   * other trees.
   */
  void softDelete();

  /**
   * Inserts a point into the tree. The point will be copied to the data matrix
   * of the leaf node where it is finally inserted, but we pass by reference since
   * it may be passed many times before it actually reaches a leaf.
   * @param point The point (arma::vec&) to be inserted.
   */
  void InsertPoint(const arma::vec& point);

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
  const HRectBound& Bound() const { return bound; }
  //! Modify the bound object for this node.
  HRectBound& Bound() { return bound; }

  //! Return the statistic object for this node.
  const StatisticType& Stat() const { return stat; }
  //! Modify the statistic object for this node.
  StatisticType& Stat() { return stat; }

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
  const arma::mat& Dataset() const { return dataset; }
  //! Modify the dataset which the tree is built on.  Be careful!
  arma::mat& Dataset() { return dataset; }

  //! Get the metric which the tree uses.
  typename BoundType::MetricType Metric() const { return bound.Metric(); }

  //! Get the centroid of the node and store it in the given vector.
  void Centroid(arma::vec& centroid) { bound.Centroid(centroid); }

  //! Return the number of child nodes.  (One level beneath this one only.)
  size_t NumChildren() const { return numChildren; }
  //! Modify the number of child nodes.  Be careful.
  size_t& NumChildren() { return numChildren; }

  //! Get the children of this node.
  const std::vector<RectangleTree*>& Children() const { return children; }
  //! Modify the children of this node.
  std::vector<RectangleTree*>& Children() { return children; }

  /**
   * Return the furthest distance to a point held in this node.  If this is not
   * a leaf node, then the distance is 0 because the node holds no points.
   */
  double FurthestPointDistance() const;

  /**
   * Return the furthest possible descendant distance.  This returns the maximum
   * distance from the centroid to the edge of the bound and not the empirical
   * quantity which is the actual furthest descendant distance.  So the actual
   * furthest descendant distance may be less than what this method returns (but
   * it will never be greater than this).
   */
  double FurthestDescendantDistance() const;

  //! Return the distance from the center of this node to the center of the
  //! parent node.
  double ParentDistance() const { return parentDistance; }
  //! Modify the distance from the center of this node to the center of the
  //! parent node.
  double& ParentDistance() { return parentDistance; }

  /**
   * Return the specified child.
   *
   * @param child Index of child to return.
   */
  RectangleTree& Child(const size_t child) const;

  //! Return the number of points in this node (returns 0 if this node is not a leaf).
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
  double MinDistance(const RectangleTree* other) const
  {
    return bound.MinDistance(other->Bound());
  }

  //! Return the maximum distance to another node.
  double MaxDistance(const RectangleTree* other) const
  {
    return bound.MaxDistance(other->Bound());
  }

  //! Return the minimum and maximum distance to another node.
  math::Range RangeDistance(const RectangleTree* other) const
  {
    return bound.RangeDistance(other->Bound());
  }


  //! Return the minimum distance to another point.
  template<typename VecType>
  double MinDistance(const VecType& point,
                     typename boost::enable_if<IsVector<VecType> >::type* = 0)
      const
  {
    return bound.MinDistance(point);
  }

  //! Return the maximum distance to another point.
  template<typename VecType>
  double MaxDistance(const VecType& point,
                     typename boost::enable_if<IsVector<VecType> >::type* = 0)
      const
  {
    return bound.MaxDistance(point);
  }

  //! Return the minimum and maximum distance to another point.
  template<typename VecType>
  math::Range
  RangeDistance(const VecType& point,
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

  /**
   * Gets the index one beyond the last index in the subset.  CURRENTLY MEANINGLESS!
   */
  size_t End() const;

  //! Return the number of points in this subset.
  size_t Count() const { return count; }
  //! Modify the number of points in this subset.
  size_t& Count() { return count; }

  //! Returns false: this tree type does not have self children.
  static bool HasSelfChildren() { return false; }

 private:
  /**
   * Private copy constructor, available only to fill (pad) the tree to a
   * specified level.
   */
  RectangleTree(const size_t begin,
                  const size_t count,
                  HRectBound bound,
                  StatisticType stat,
                  const int maxLeafSize = 20) :
      begin(begin),
      count(count),
      bound(bound),
      stat(stat),
      maxLeafSize(maxLeafSize) { }

  RectangleTree* CopyMe()
  {
    return new RectangleTree(begin, count, bound, stat, maxLeafSize);
  }

  /**
   * Splits the current node, recursing up the tree.
   *
   * @param tree The RectangleTree object (node) to split.
   */
  void SplitNode(RectangleTree& tree);

  /**
   * Splits the current node, recursing up the tree.
   * CURRENTLY IT DOES NOT Also returns a list of the changed indices (because there are none).
   *
   * @param data Dataset which we are using.
   * @param oldFromNew Vector holding permuted indices NOT IMPLEMENTED.
   */
  void SplitNode(MatType& data, std::vector<size_t>& oldFromNew);

 public:
  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;

};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "rectangle_tree_impl.hpp"

#endif
