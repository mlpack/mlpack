/**
 * @file minimal_splits_number_sweep.hpp
 * @author Mikhail Lozhnikov
 *
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_HPP

namespace mlpack {
namespace tree {

template<typename SplitPolicy>
class MinimalSplitsNumberSweep
{
 private:
  template<typename ElemType>
  struct SortStruct
  {
    ElemType d;
    int n;
  };

  template<typename ElemType>
  static bool StructComp(const SortStruct<ElemType>& s1,
                         const SortStruct<ElemType>& s2)
  {
    return s1.d < s2.d;
  }
 public:
  template<typename>
  struct SweepCost
  {
    typedef size_t type;
  };

  template<typename TreeType>
  static size_t SweepNonLeafNode(size_t axis, const TreeType* node,
      typename TreeType::ElemType& axisCut);

  template<typename TreeType>
  static size_t SweepLeafNode(size_t axis, const TreeType* node,
      typename TreeType::ElemType& axisCut);
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "minimal_splits_number_sweep_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_HPP


