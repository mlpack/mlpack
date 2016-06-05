/**
 * @file recursive_hilbert_value.hpp
 * @author Mikhail Lozhnikov
 *
 * Defintion of the RecursiveHilbertValue class, a class that measures
 * ordering of points recursively.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RECURSIVE_HILBERT_VALUE_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_RECURSIVE_HILBERT_VALUE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

constexpr int recursionDepth = 500;

template<typename TreeElemType>
class RecursiveHilbertValue
{
 public:
  //! Default constructor
  RecursiveHilbertValue();

  /**
   * Construct this for the node tree. If the node is the root this method
   * computes the Hilbert value for each point in the tree's dataset.
   * @param node The node that stores this Hilbert value.
   */
  template<typename TreeType>
  RecursiveHilbertValue(const TreeType* tree);

  /**
   * Create a Hilbert value object by copying from another one.
   * @param other The Hilbert value object from which the value will be copied.
   */
  RecursiveHilbertValue(const RecursiveHilbertValue& other);

  ~RecursiveHilbertValue();

  //! This struct is designed in order to facilitate the recursion.
  typedef struct tagCompareStruct
  {
    //! Lower bound
    arma::Col<TreeElemType> Lo;
    //! High bound
    arma::Col<TreeElemType> Hi;
    //! Permutation of axes
    std::vector<size_t> permutation;
    //! Indicates that the axis should be inverted
    std::vector<bool> inversion;
    //! Indicates that the result should be inverted
    arma::Col<TreeElemType> center;
    arma::Col<TreeElemType> vec;
    std::vector<int> bits;
    std::vector<int> bits2;
    bool invertResult;
    int recursionLevel;


    tagCompareStruct(size_t dim) :
      Lo(dim),
      Hi(dim),
      permutation(dim),
      inversion(dim),
      center(dim),
      vec(dim),
      bits(dim),
      bits2(dim),
      invertResult(false),
      recursionLevel(0)
    {
      for(size_t i = 0; i < dim; i++)
      {
        Lo[i] = std::numeric_limits<TreeElemType>::lowest();
        Hi[i] = std::numeric_limits<TreeElemType>::max();
        permutation[i] = i;
        inversion[i] = false;
      }
    }
  } CompareStruct;

  /**
   * Compare two points. It returns 1 if the first point is greater than
   * the second one, -1 if the first point is less than the second one and
   * 0 if the Hilbert values of the points are equal.
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
   * 0 if the values are equal.
   * @param val1 The first Hilbert value.
   * @param val2 The second Hilbert value.
   */

  static int CompareValues(const RecursiveHilbertValue& val1,
                           const RecursiveHilbertValue& val2);

  /**
   * Compare the largest Hilbert value of the node with the val value.
   * It returns 1 if the value of the node is greater than val,
   * -1 if the value of the node is less than val and
   * 0 if the values are equal.
   * @param val The Hilbert value to compare with.
   */
  int CompareWith(const RecursiveHilbertValue& val) const;

  /**
   * Compare the largest Hilbert value of the node with the Hilbert value
   * of the point. It returns 1 if the value of the node is greater than
   * the value of the point, -1 if the value of the node is less than
   * the value of the point and 0 if the values are equal.
   * @param point The point to compare with.
   */
  template<typename VecType>
  int CompareWith(const VecType& point,
                  typename boost::enable_if<IsVector<VecType>>* = 0) const;

  template<typename VecType>
  int CompareWithCachedPoint(const VecType& point,
                  typename boost::enable_if<IsVector<VecType>>* = 0) const;


  /**
   * Update the largest Hilbert value of the node.
   * @param node The node in which the point is being inserted.
   * @param point The number of the point being inserted.
   */
  template<typename TreeType, typename VecType>
  size_t InsertPoint(TreeType* node, const VecType& point,
                             typename boost::enable_if<IsVector<VecType>>* = 0);

  /**
   * Update the largest Hilbert value of the node.
   * @param node The node being inserted.
   */
  template<typename TreeType>
  void InsertNode(TreeType* node);

  /**
   * Update the largest Hilbert value of the node.
   * @param node The node from which another node is being deleted.
   * @param nodeIndex The number of the node being deleted.
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
   * Copy the largest Hilbert value.
   * @param dst The node to which the information is being copied.
   * @param src The node from which the information is being copied.
   */
  template<typename TreeType>
  void Copy(TreeType* dst, TreeType* src);

  void NullifyData();

  /**
   * Update the largest Hilbert value.
   * @param node The node in which the information should be updated.
   */
  template<typename TreeType>
  void UpdateLargestValue(TreeType* node);

  template<typename TreeType>
  void UpdateHilbertValues(TreeType* parent, size_t firstSibling,
                           size_t lastSibling);

  //! Return the largest Hilbert value
  const arma::Col<TreeElemType>* LargestValue() const { return largestValue; }

  //! Modify the largest Hilbert value
  arma::Col<TreeElemType>*& LargestValue() { return largestValue; }

 private:
  //! The point that has the largest Hilbert value.
  arma::Col<TreeElemType>* largestValue;
  bool ownsLargestValue;
  bool hasLargestValue;

  /**
   * Compare two points. It returns 1 if the first point is greater than
   * the second one, -1 if the first point is less than the second one and
   * 0 if the Hilbert values of the points are equal.
   * @param pt1 The first point.
   * @param pt2 The second point.
   * @param comp An object of CompareStruct.
   */
  template<typename VecType1, typename VecType2>
  static int ComparePoints(const VecType1& pt1, const VecType2& pt2,
       CompareStruct& comp, typename boost::enable_if<IsVector<VecType1>>* = 0,
                            typename boost::enable_if<IsVector<VecType2>>* = 0);
 public:
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);
};
} // namespace tree
} // namespace mlpack

// Include implementation
#include "recursive_hilbert_value_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_RECURSIVE_HILBERT_VALUE_HPP
