/**
 * @file discrete_hilbert_value.hpp
 * @author Mikhail Lozhnikov
 *
 * Defintion of the DiscreteHilbertValue class, a class that calculates
 * the ordering of points using the Hilbert curve.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

template<typename TreeElemType>
class DiscreteHilbertValue
{
 public:
  typedef typename std::conditional<sizeof(TreeElemType)*CHAR_BIT <= 32,
                                    uint32_t,
                                    uint64_t>::type HilbertElemType;
  //! Default constructor
  DiscreteHilbertValue();

  /**
   * Construct this for the node tree. If the node is the root this method
   * computes the Hilbert value for each point in the tree's dataset.
   * @param node The node that stores this Hilbert value.
   */
  template<typename TreeType>
  DiscreteHilbertValue(const TreeType* tree);

  /**
   * Create a Hilbert value object by copying from another one.
   * @param other The Hilbert value object from which the value will be copied.
   */
  DiscreteHilbertValue(const DiscreteHilbertValue& other);

  //! Free memory
  ~DiscreteHilbertValue();

  /**
   * Compare two points. It returns 1 if the first point is greater than
   * the second one, -1 if the first point is less than the second one and
   * 0 if the Hilbert values of the points are equal. In order to do it
   * this method computes the Hilbert values of the points.
   * @param pt1 The first point.
   * @param pt2 The second point.
   */
  template<typename VecType1, typename VecType2>
  static int ComparePoints(const VecType1& pt1, const VecType2& pt2,
                           typename boost::enable_if<IsVector<VecType1>>* = 0,
                           typename boost::enable_if<IsVector<VecType2>>* = 0);

  /**
   * Compare two Hilbert values. It returns 1 if the first value is greater than
   * the second one, -1 if the first value is less than the second one and
   * 0 if the values are equal. This method does not compute the Hilbert values.
   * @param val1 The first point.
   * @param val2 The second point.
   */
  static int CompareValues(const DiscreteHilbertValue& val1,
                                              const DiscreteHilbertValue& val2);

  /**
   * Compare the largest Hilbert value of the node with the val value.
   * It returns 1 if the value of the node is greater than val,
   * -1 if the value of the node is less than val and
   * 0 if the values are equal. This method does not compute the Hilbert values.
   * @param val The Hilbert value to compare with.
   */
  int CompareWith(const DiscreteHilbertValue& val) const;

  /**
   * Compare the largest Hilbert value of the node with the Hilbert value
   * of the point. It returns 1 if the value of the node is greater than
   * the value of the point, -1 if the value of the node is less than
   * the value of the point and 0 if the values are equal.
   * This method computes the Hilbert value of the point.
   * @param tree Not used
   * @param val The point to compare with.
   */
  template<typename VecType>
  int CompareWith(const VecType& pt,
                  typename boost::enable_if<IsVector<VecType>>* = 0) const;

  /**
   * Compare the largest Hilbert value of the node with the Hilbert value
   * of the point. It returns 1 if the value of the node is greater than
   * the value of the point, -1 if the value of the node is less than
   * the value of the point and 0 if the values are equal.
   * This method computes the Hilbert value of the point.
   * @param tree Not used
   * @param val The number of the point to compare with.
   */

  template<typename VecType>
  int CompareWithCachedPoint(const VecType& pt,
                  typename boost::enable_if<IsVector<VecType>>* = 0) const;

  /**
   * Update the largest Hilbert value of the node and insert the point
   * in the local dataset if the node is a leaf.
   * @param node The node in which the point is being inserted.
   * @param point The number of the point being inserted.
   */
  template<typename TreeType, typename VecType>
  size_t InsertPoint(TreeType *node, const VecType& pt,
                     typename boost::enable_if<IsVector<VecType>>* = 0);
  /**
   * Update the largest Hilbert value of the node.
   * @param node The node being inserted.
   */
  template<typename TreeType>
  void InsertNode(TreeType* node);

  /**
   * Update the largest Hilbert value of the node and delete the point
   * from the local dataset.
   * @param node The node from which the point is being deleted.
   * @param localIndex The number of the point in the local dataset.
   */
  template<typename TreeType>
  void DeletePoint(TreeType* node, const size_t localIndex);

  /**
   * Update the largest Hilbert value of the node.
   * @param node The node from which another node is being deleted.
   * @param nodeIndex The number of the node being deleted.
   */
  template<typename TreeType>
  void RemoveNode(TreeType* node, const size_t nodeIndex);

  /**
   * Copy the largest Hilbert value and the local dataset
   * @param dst The node to which the information is being copied.
   * @param src The node from which the information is being copied.
   */
  template<typename TreeType>
  void Copy(TreeType* dst, TreeType* src);
  
  void NullifyData();
  
  /**
   * Update the largest Hilbert value and the local dataset.
   * The children of the node (or the points that the node contains) should be
   * arranged according to their Hilbert values.
   * @param node The node in which the information should be updated.
   */
  template<typename TreeType>
  void UpdateLargestValue(TreeType* node);

  template<typename TreeType>
  void UpdateHilbertValues(TreeType* parent, size_t firstSibling,
                           size_t lastSibling);

  //! Return the number of values
  size_t NumValues() const
  { return numValues; }

  //! Modify the number of values
  size_t& NumValues()
  { return numValues; }

  //! Return the local dataset
  const arma::Mat<HilbertElemType>* LocalDataset() const
  { return localDataset; }

  //! Modify the dataset
  arma::Mat<HilbertElemType>*& LocalDataset() { return localDataset; }
  
  //! Modify the valueToInsert
  arma::Col<HilbertElemType>* ValueToInsert() { return valueToInsert; }

  //! Modify the valueToInsert
  const arma::Col<HilbertElemType>* ValueToInsert() const
  { return valueToInsert; }

 private:
  //! The number of bits that we can store
  static constexpr size_t order = sizeof(HilbertElemType) * CHAR_BIT;
  //! The local dataset
  arma::Mat<HilbertElemType>* localDataset;
  //! Indicates that the node owns the local dataset
  bool ownsLocalDataset;
  //! The number of values in the local dataset
  size_t numValues;
  //! The Hilbert value of the point that is being inserted
  arma::Col<HilbertElemType>* valueToInsert;
  //! Indicates that the node owns the valueToInsert 
  bool ownsValueToInsert;

  /**
   * Calculate the Hilbert value of the point pt.
   * @param pt The point for which the Hilbert value should be calculated.
   */
  template<typename VecType>
  static arma::Col<HilbertElemType> CalculateValue(const VecType& pt,
                             typename boost::enable_if<IsVector<VecType>>* = 0);

  /**
   * Compare two Hilbert values. It returns 1 if the first value is greater than
   * the second one, -1 if the first value is less than the second one and
   * 0 if the values are equal. This method does not compute the Hilbert values.
   * @param value1 The first value.
   * @param value2 The second value.
   */
  static int CompareValues(const arma::Col<HilbertElemType>& value1,
                           const arma::Col<HilbertElemType>& value2);
  /**
   * Returns true if the node has the largest Hilbert value.
   */
  bool HasValue() const;
};
} // namespace tree
} // namespace mlpack

// Include implementation
#include "discrete_hilbert_value_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
