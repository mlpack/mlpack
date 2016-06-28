/**
 * @file minimal_splits_number_sweep.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the MinimalSplitsNumberSweep class, a class that finds a
 * partition of a node along an axis.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_HPP

namespace mlpack {
namespace tree {

/**
 * The MinimalSplitsNumberSweep class finds a partition along which we
 * can split a node according to the number of required splits of the node.
 * Moreover, the class evaluates the cost of each split.
 *
 * @tparam SplitPolicy The class that provides rules for inserting children of
 *    a node that is being split into two new subtrees.
 */
template<typename SplitPolicy>
class MinimalSplitsNumberSweep
{
 private:
  /**
   * Class to allow for faster sorting.
   */
  template<typename ElemType>
  struct SortStruct
  {
    ElemType d;
    int n;
  };

  /**
   * Comparator for sorting with SortStruct.
   */
  template<typename ElemType>
  static bool StructComp(const SortStruct<ElemType>& s1,
                         const SortStruct<ElemType>& s2)
  {
    return s1.d < s2.d;
  }
 public:
  //! A struct that provides the type of the sweep cost.
  template<typename>
  struct SweepCost
  {
    typedef size_t type;
  };

  /**
   * Find a suitable partition of a non-leaf node along the provided axis.
   * The method returns the cost of the split.
   *
   * @param axis The axis along which we are finding a partition.
   * @param node The node that is being split.
   * @param axisCut The coordinate at which the node may be split.
   */
  template<typename TreeType>
  static size_t SweepNonLeafNode(
      const size_t axis,
      const TreeType* node,
      typename TreeType::ElemType& axisCut);

  /**
   * Find a suitable partition of a leaf node along the provided axis.
   * The method returns the cost of the split.
   *
   * @param axis The axis along which we are finding a partition.
   * @param node The node that is being split.
   * @param axisCut The coordinate at which the node may be split.
   */
  template<typename TreeType>
  static size_t SweepLeafNode(
      const size_t axis,
      const TreeType* node,
      typename TreeType::ElemType& axisCut);
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "minimal_splits_number_sweep_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_HPP


