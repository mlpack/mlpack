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

namespace mlpack {
namespace tree {

template<typename MetricType,
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
  HRectBound<MeetricType> bound;
  //! The dataset.
  MatType* dataset;
  //! The parent (NULL if this node is the root).
  Octree* parent;

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
  Octree(const MatType& data, const size_t maxLeafSize = 20);

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

 private:
  /**
   * Split the node, using the given center and the given maximum width of this
   * node.
   *
   * @param center Center of the node.
   * @param width Width of the current node.
   */
  void SplitNode(const arma::vec& center, const double width);

  /**
   * Split the node, using the given center and the given maximum width of this
   * node, and fill the mappings vector.
   *
   * @param center Center of the node.
   * @param width Width of the current node.
   * @param oldFromNew Mappings from old to new.
   */
  void SplitNode(const arma::vec& center,
                 const double width,
                 std::vector<size_t>& oldFromNew);
};
