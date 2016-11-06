/**
 * @file minimal_coverage_sweep.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the MinimalCoverageSweep class, a class that finds a partition
 * of a node along an axis.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_HPP

namespace mlpack {
namespace tree {

/**
 * The MinimalCoverageSweep class finds a partition along which we
 * can split a node according to the coverage of two resulting nodes. The class
 * finds a partition along a given axis. Moreover, the class evaluates the cost
 * of each split. The cost is proportional to the total coverage of resulting
 * nodes. If the resulting nodes are overflowed the maximum cost is returned.
 *
 * @tparam SplitPolicy The class that provides rules for inserting children of
 *    a node that is being split into two new subtrees.
 */
template<typename SplitPolicy>
class MinimalCoverageSweep
{
 public:
  //! A struct that provides the type of the sweep cost.
  template<typename TreeType>
  struct SweepCost
  {
    typedef typename TreeType::ElemType type;
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
  static typename TreeType::ElemType SweepNonLeafNode(
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
  static typename TreeType::ElemType SweepLeafNode(
      const size_t axis,
      const TreeType* node,
      typename TreeType::ElemType& axisCut);

  /**
   * Check if an intermediate node can be split along the axis at the provided
   * coordinate.
   *
   * @param node The node that is being split.
   * @param cutAxis The axis that we want to check.
   * @param cut The coordinate that we want to check.
   */
  template<typename TreeType, typename ElemType>
  static bool CheckNonLeafSweep(const TreeType* node,
                                const size_t cutAxis,
                                const ElemType cut);

  /**
   * Check if a leaf node can be split along the axis at the provided
   * coordinate.
   *
   * @param node The node that is being split.
   * @param cutAxis The axis that we want to check.
   * @param cut The coordinate that we want to check.
   */
  template<typename TreeType, typename ElemType>
  static bool CheckLeafSweep(const TreeType* node,
                             const size_t cutAxis,
                             const ElemType cut);
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "minimal_coverage_sweep_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_HPP

