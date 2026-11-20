/**
 * @file core/tree/rectangle_tree/discrete_hilbert_value.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the DiscreteHilbertValue class, a class that calculates
 * the ordering of points using the Hilbert curve.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The DiscreteHilbertValue class stores Hilbert values for all of the points in
 * a RectangleTree node, and calculates Hilbert values for new points.  This
 * implementation calculates the full discrete Hilbert value; for a
 * d-dimensional vector filled with elements of size E, each Hilbert value will
 * take dE space.
 */
template<typename TreeElemType>
class DiscreteHilbertValue
{
 public:
  //! Depending on the precision of the tree element type, we may need to use
  //! uint32_t or uint64_t.
  using HilbertElemType = std::conditional_t<
      (sizeof(TreeElemType) * CHAR_BIT <= 32), uint32_t, uint64_t>;

  //! Default constructor.
  DiscreteHilbertValue();

  /**
   * Construct this for the node tree. If the node is the root this method
   * computes the Hilbert value for each point in the tree's dataset.
   *
   * @param node The node that stores this Hilbert value.
   */
  template<typename TreeType>
  DiscreteHilbertValue(const TreeType* tree);

  /**
   * Create a Hilbert value object by copying from another one.
   *
   * @param other The object from which the value will be copied.
   * @param tree The node that holds the Hilbert value.
   * @param deepCopy If false, the dataset will not be copied.
   */
  template<typename TreeType>
  DiscreteHilbertValue(const DiscreteHilbertValue& other,
                       TreeType* tree,
                       bool deepCopy);

  /**
   * Create a Hilbert value object by moving another one.
   *
   * @param other The Hilbert value object from which the value will be moved.
   */
  DiscreteHilbertValue(DiscreteHilbertValue&& other);

  //! Free memory
  ~DiscreteHilbertValue();

  /**
   * Compare two points. It returns 1 if the first point is greater than the
   * second one, -1 if the first point is less than the second one and 0 if the
   * Hilbert values of the points are equal. In order to do it this method
   * computes the Hilbert values of the points.
   *
   * @param pt1 The first point.
   * @param pt2 The second point.
   */
  template<typename VecType1, typename VecType2>
  static int ComparePoints(
      const VecType1& pt1,
      const VecType2& pt2,
      typename std::enable_if_t<IsVector<VecType1>::value>* = 0,
      typename std::enable_if_t<IsVector<VecType2>::value>* = 0);

  /**
   * Compare two Hilbert values. It returns 1 if the first value is greater than
   * the second one, -1 if the first value is less than the second one and 0 if
   * the values are equal. This method does not compute the Hilbert values.
   *
   * @param val1 The first point.
   * @param val2 The second point.
   */
  static int CompareValues(const DiscreteHilbertValue& val1,
                           const DiscreteHilbertValue& val2);

  /**
   * Compare the largest Hilbert value of the node with the val value.  It
   * returns 1 if the value of the node is greater than val, -1 if the value of
   * the node is less than val and 0 if the values are equal. This method does
   * not compute the Hilbert values.
   *
   * @param val The Hilbert value to compare with.
   */
  int CompareWith(const DiscreteHilbertValue& val) const;

  /**
   * Compare the largest Hilbert value of the node with the Hilbert value of the
   * point. It returns 1 if the value of the node is greater than the value of
   * the point, -1 if the value of the node is less than the value of the point
   * and 0 if the values are equal.  This method computes the Hilbert value of
   * the point.
   *
   * @param pt The point to compare with.
   */
  template<typename VecType>
  int CompareWith(
      const VecType& pt,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;

  /**
   * Compare the Hilbert value of the cached point with the Hilbert value of the
   * given point. It returns 1 if the value of the node is greater than the
   * value of the point, -1 if the value of the node is less than the value of
   * the point and 0 if the values are equal.  This method computes the Hilbert
   * value of the point.
   *
   * @param pt The point to compare with.
   */

  template<typename VecType>
  int CompareWithCachedPoint(
      const VecType& pt,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;

  /**
   * Update the largest Hilbert value of the node and insert the point in the
   * local dataset if the node is a leaf.
   *
   * @param node The node in which the point is being inserted.
   * @param point The number of the point being inserted.
   */
  template<typename TreeType, typename VecType>
  size_t InsertPoint(TreeType *node,
                     const VecType& pt,
                     typename std::enable_if_t<IsVector<VecType>::value>* = 0);

  /**
   * Update the largest Hilbert value of the node.
   *
   * @param node The node being inserted.
   */
  template<typename TreeType>
  void InsertNode(TreeType* node);

  /**
   * Update the largest Hilbert value of the node and delete the point from the
   * local dataset.
   *
   * @param node The node from which the point is being deleted.
   * @param localIndex The index of the point in the local dataset.
   */
  template<typename TreeType>
  void DeletePoint(TreeType* node, const size_t localIndex);

  /**
   * Update the largest Hilbert value of the node.
   *
   * @param node The node from which another node is being deleted.
   * @param nodeIndex The index of the node being deleted.
   */
  template<typename TreeType>
  void RemoveNode(TreeType* node, const size_t nodeIndex);

  /**
   * Copy the local Hilbert value's pointer.
   *
   * @param other The DiscreteHilbertValue object from which the dataset
   *    will be copied.
   */
  DiscreteHilbertValue& operator=(const DiscreteHilbertValue& other);

  /**
   * Move the local Hilbert object.
   *
   * @param other The DiscreteHilbertValue object from which the dataset
   *    will be copied.
   */
  DiscreteHilbertValue& operator=(DiscreteHilbertValue&& other);

  /**
   * Nullify the localHilbertValues pointer in order to prevent an invalid free.
   */
  void NullifyData();

  /**
   * Update the largest Hilbert value and the local Hilbert values of an
   * intermediate node.  The children of the node (or the points that the node
   * contains) should be arranged according to their Hilbert values.
   *
   * @param node The node in which the information should be updated.
   */
  template<typename TreeType>
  void UpdateLargestValue(TreeType* node);

  /**
   * This method updates the largest Hilbert value of a leaf node and
   * redistributes the Hilbert values of points according to their new position
   * after the split algorithm.
   *
   * @param parent The parent of the node that was split.
   * @param firstSibling The first cooperating sibling.
   * @param lastSibling The last cooperating sibling.
   */
  template<typename TreeType>
  void RedistributeHilbertValues(TreeType* parent,
                                 const size_t firstSibling,
                                 const size_t lastSibling);

  /**
   * Calculate the Hilbert value of the point pt.
   *
   * @param pt The point for which the Hilbert value should be calculated.
   */
  template<typename VecType>
  static arma::Col<HilbertElemType> CalculateValue(
      const VecType& pt,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0);

  /**
   * Compare two Hilbert values. It returns 1 if the first value is greater than
   * the second one, -1 if the first value is less than the second one and 0 if
   * the values are equal. This method does not compute the Hilbert values.
   *
   * @param value1 The first value.
   * @param value2 The second value.
   */
  static int CompareValues(const arma::Col<HilbertElemType>& value1,
                           const arma::Col<HilbertElemType>& value2);

  //! Return the number of values.
  size_t NumValues() const { return numValues; }
  //! Modify the number of values.
  size_t& NumValues() { return numValues; }

  //! Return the Hilbert values.
  const arma::Mat<HilbertElemType>* LocalHilbertValues() const
  { return localHilbertValues; }
  //! Modify the Hilbert values.
  arma::Mat<HilbertElemType>*& LocalHilbertValues()
  { return localHilbertValues; }

  //! Return the ownsLocalHilbertValues variable.
  bool OwnsLocalHilbertValues() const { return ownsLocalHilbertValues; }
  //! Modify the ownsLocalHilbertValues variable.
  bool& OwnsLocalHilbertValues() { return ownsLocalHilbertValues; }

  //! Return the cached point (valueToInsert).
  const arma::Col<HilbertElemType>* ValueToInsert() const
  { return valueToInsert; }
  //! Modify the cached point (valueToInsert).
  arma::Col<HilbertElemType>* ValueToInsert() { return valueToInsert; }

  //! Return the ownsValueToInsert variable.
  bool OwnsValueToInsert() const { return ownsValueToInsert; }
  //! Modify the ownsValueToInsert variable.
  bool& OwnsValueToInsert() { return ownsValueToInsert; }
 private:
  //! The number of bits that we can store.
  static constexpr size_t order = sizeof(HilbertElemType) * CHAR_BIT;
  //! The local Hilbert values.
  arma::Mat<HilbertElemType>* localHilbertValues;
  //! Indicates that the node owns the localHilbertValues variable.
  bool ownsLocalHilbertValues;
  //! The number of values in the localHilbertValues dataset.
  size_t numValues;
  /** The Hilbert value of the point that is being inserted.
   * The pointer is the same in all nodes. The value is updated in InsertPoint()
   * if it is invoked at the root level. This variable helps to avoid
   * multiple computation of the Hilbert value of a point in the insertion
   * process.
   */
  arma::Col<HilbertElemType>* valueToInsert;
  //! Indicates that the node owns the valueToInsert.
  bool ownsValueToInsert;

 public:
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
};

} // namespace mlpack

// Include implementation.
#include "discrete_hilbert_value_impl.hpp"

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
