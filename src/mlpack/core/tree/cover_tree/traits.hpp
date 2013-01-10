/**
 * @file traits.hpp
 * @author Ryan Curtin
 *
 * This file contains the specialization of the TreeTraits class for the
 * CoverTree type of tree.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_TRAITS_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {
namespace tree {

/**
 * The specialization of the TreeTraits class for the CoverTree tree type.  It
 * defines characteristics of the cover tree, and is used to help write
 * tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
template<>
template<typename MetricType,
         typename RootPointPolicy,
         typename StatisticType>
class TreeTraits<CoverTree<MetricType, RootPointPolicy, StatisticType> >
{
 public:
  /**
   * The cover tree calculates the distance between parent and child during
   * construction, so that value is saved and CoverTree<...>::ParentDistance()
   * does exist.
   */
  static const bool HasParentDistance = true;

  /**
   * The cover tree (or, this implementation of it) does not require that
   * children represent non-overlapping subsets of the parent node.
   */
  static const bool HasOverlappingChildren = true;
};

}; // namespace tree
}; // namespace mlpack

#endif
