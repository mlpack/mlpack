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
class RecursiveHilbertValue
{
 public:
  //! Default constructor
  RecursiveHilbertValue() :
    largestValue(-1)
  { };

  /**
   * Construct this for the node tree. If the node is the root this method
   * computes the Hilbert value for each point in the tree's dataset.
   * @param node The node that stores this Hilbert value.
   */
  template<typename TreeType>
  RecursiveHilbertValue(const TreeType *) :
    largestValue(-1)
  { };

  /**
   * Create a Hilbert value object by copying from the other node.
   * @param other The node from which the value will be copied.
   */
  template<typename TreeType>
  RecursiveHilbertValue(const TreeType &other) :
    largestValue(other.AuxiliaryInfo().LargestHilbertValue().LargestValue())
  { };

  //! This struct is designed in order to facilitate the recursion.
  template<typename ElemType>
  struct tagCompareStruct
  {
    //! Lower bound
    arma::Col<ElemType> Lo;
    //! High bound
    arma::Col<ElemType> Hi;
    //! Permutation of axes
    std::vector<size_t> permutation;
    //! Indicates that the axis should be inverted
    std::vector<bool> inversion;
    //! Indicates that the result should be inverted
    arma::Col<ElemType> center;
    arma::Col<ElemType> vec;
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
        Lo[i] = std::numeric_limits<ElemType>::lowest();
        Hi[i] = std::numeric_limits<ElemType>::max();
        permutation[i] = i;
        inversion[i] = false;
      }
    }
  };
  template<typename ElemType>
  using CompareStruct = struct tagCompareStruct<ElemType>;

  /**
   * Compare two points. It returns 1 if the first point is greater than
   * the second one, -1 if the first point is less than the second one and
   * 0 if the Hilbert values of the points are equal.
   * @param pt1 The first point.
   * @param pt2 The second point.
   */
  template<typename ElemType>
  static int ComparePoints(const arma::Col<ElemType> &pt1,
                                                const arma::Col<ElemType> &pt2);

  /**
   * Compare two Hilbert values. It returns 1 if the first value is greater than
   * the second one, -1 if the first value is less than the second one and
   * 0 if the values are equal.
   * @param val1 The first Hilbert value.
   * @param val2 The second Hilbert value.
   */
  template<typename TreeType>
  static int CompareValues(TreeType *tree, RecursiveHilbertValue &val1,
                                                   RecursiveHilbertValue &val2);

  /**
   * Compare the largest Hilbert value of the node with the val value.
   * It returns 1 if the value of the node is greater than val,
   * -1 if the value of the node is less than val and
   * 0 if the values are equal.
   * @param tree The pointer to the tree.
   * @param val The Hilbert value to compare with.
   */
  template<typename TreeType>
  int CompareWith(TreeType *tree, RecursiveHilbertValue &val);

  /**
   * Compare the largest Hilbert value of the node with the Hilbert value
   * of the point. It returns 1 if the value of the node is greater than
   * the value of the point, -1 if the value of the node is less than
   * the value of the point and 0 if the values are equal.
   * @param tree The pointer to the tree.
   * @param pt The point to compare with.
   */
  template<typename TreeType,typename ElemType>
  int CompareWith(TreeType *tree, const arma::Col<ElemType> &pt);

  /**
   * Compare the largest Hilbert value of the node with the Hilbert value
   * of the point. It returns 1 if the value of the node is greater than
   * the value of the point, -1 if the value of the node is less than
   * the value of the point and 0 if the values are equal.
   * @param tree The pointer to the tree.
   * @param point The number of the point to compare with.
   */
  template<typename TreeType>
  int CompareWith(TreeType *tree, const size_t point);

  /**
   * Update the largest Hilbert value of the node.
   * @param node The node in which the point is being inserted.
   * @param point The number of the point being inserted.
   */
  template<typename TreeType>
  size_t InsertPoint(TreeType *node, const size_t point);

  /**
   * Update the largest Hilbert value of the node.
   * @param node The node being inserted.
   */
  template<typename TreeType>
  void InsertNode(TreeType *node);

  /**
   * Update the largest Hilbert value of the node.
   * @param node The node from which another node is being deleted.
   * @param nodeIndex The number of the node being deleted.
   */
  template<typename TreeType>
  void DeletePoint(TreeType *node, const size_t localIndex);

  /**
   * Update the largest Hilbert value of the node.
   * @param node The node from which another node is being deleted.
   * @param nodeIndex The number of the node being deleted.
   */
  template<typename TreeType>
  void RemoveNode(TreeType *node, const size_t nodeIndex);

  /**
   * Copy the largest Hilbert value.
   * @param dst The node to which the information is being copied.
   * @param src The node from which the information is being copied.
   */
  RecursiveHilbertValue operator = (const RecursiveHilbertValue &val);

  /**
   * Copy the largest Hilbert value.
   * @param dst The node to which the information is being copied.
   * @param src The node from which the information is being copied.
   */
  template<typename TreeType>
  void Copy(TreeType *dst, TreeType *src);

  /**
   * Update the largest Hilbert value.
   * @param node The node in which the information should be updated.
   */
  template<typename TreeType>
  void UpdateLargestValue(TreeType *node);

  //! Return the largest Hilbert value
  ptrdiff_t LargestValue() const { return largestValue; }

  //! Modify the largest Hilbert value
  ptrdiff_t& LargestValue() { return largestValue; }

 private:
  //! The largest Hilbert value i.e. the number of the point in the dataset.
  ptrdiff_t largestValue;

  /**
   * Compare two points. It returns 1 if the first point is greater than
   * the second one, -1 if the first point is less than the second one and
   * 0 if the Hilbert values of the points are equal.
   * @param pt1 The first point.
   * @param pt2 The second point.
   * @param comp An object of CompareStruct.
   */
  template<typename ElemType>
  static int ComparePoints(const arma::Col<ElemType> &pt1,
                           const arma::Col<ElemType> &pt2,
                           CompareStruct<ElemType> &comp);

};
} // namespace tree
} // namespace mlpack

// Include implementation
#include "recursive_hilbert_value_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_RECURSIVE_HILBERT_VALUE_HPP
