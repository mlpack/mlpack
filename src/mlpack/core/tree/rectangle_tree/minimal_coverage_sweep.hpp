/**
 * @file minimal_coverage_sweep.hpp
 * @author Mikhail Lozhnikov
 *
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_HPP

namespace mlpack {
namespace tree {

constexpr double fillFactor = 0.5;

template<typename SplitPolicy>
class MinimalCoverageSweep
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

  template<typename TreeType>
  struct SweepCost
  {
    typedef typename TreeType::ElemType type;
  };

  template<typename TreeType>
  static typename TreeType::ElemType SweepNonLeafNode(size_t axis,
      const TreeType* node, typename TreeType::ElemType& axisCut);

  template<typename TreeType>
  static typename TreeType::ElemType SweepLeafNode(size_t axis,
      const TreeType* node, typename TreeType::ElemType& axisCut);

  template<typename TreeType, typename ElemType>
  static bool CheckNonLeafSweep(const TreeType* node, size_t cutAxis,
      ElemType cut);

  template<typename TreeType, typename ElemType>
  static bool CheckLeafSweep(const TreeType* node, size_t cutAxis,
      ElemType cut);
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "minimal_coverage_sweep_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_HPP

