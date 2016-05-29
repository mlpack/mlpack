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

class RecursiveHilbertValue
{
 public:

  RecursiveHilbertValue() :
    largestValue(-1)
  { };

  template<typename TreeType>
  RecursiveHilbertValue(const TreeType *) :
    largestValue(-1)
  { };

  template<typename TreeType>
  RecursiveHilbertValue(const TreeType &other) :
    largestValue(other.AuxiliaryInfo().LargestHilbertValue().LargestValue())
  { };

  template<typename ElemType>
  struct tagCompareStruct
  {
    arma::Col<ElemType> Lo;
    arma::Col<ElemType> Hi;
    std::vector<size_t> permutation;
    std::vector<bool> inversion;
    bool invertResult;

    tagCompareStruct(size_t dim) :
      Lo(dim),
      Hi(dim),
      permutation(dim),
      inversion(dim),
      invertResult(false)
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

  

  template<typename ElemType>
  static int ComparePoints(const arma::Col<ElemType> &pt1,
                                                const arma::Col<ElemType> &pt2);

  template<typename TreeType>
  static int CompareValues(TreeType *tree, RecursiveHilbertValue &val1,
                                                   RecursiveHilbertValue &val2);

  template<typename TreeType>
  int CompareWith(TreeType *tree, RecursiveHilbertValue &val);

  template<typename TreeType,typename ElemType>
  int CompareWith(TreeType *tree, const arma::Col<ElemType> &pt);

  template<typename TreeType>
  size_t InsertPoint(TreeType *node, const size_t point);

  template<typename TreeType>
  void InsertNode(TreeType *node);

  template<typename TreeType>
  void DeletePoint(TreeType *node, const size_t localIndex);

  template<typename TreeType>
  void RemoveNode(TreeType *node, const size_t nodeIndex);

  RecursiveHilbertValue operator = (const RecursiveHilbertValue &val);

  template<typename TreeType>
  void Copy(TreeType *dst, TreeType *src);

  size_t LargestValue() const { return largestValue; }

 private:

  ptrdiff_t largestValue;

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
