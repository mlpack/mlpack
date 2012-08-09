/**
 * @file cover_tree_map_entry.hpp
 * @author Ryan Curtin
 *
 * Definition of a simple struct which is used in cover tree traversal to
 * represent the data associated with a single cover tree node.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_MAP_ENTRY_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_MAP_ENTRY_HPP

namespace mlpack {
namespace tree {

//! This is the structure the cover tree map will use for traversal.
template<typename MetricType, typename RootPointPolicy, typename StatisticType>
struct CoverTreeMapEntry
{
  //! The node this entry refers to.
  CoverTree<MetricType, RootPointPolicy, StatisticType>* node;
  //! The score of the node.
  double score;
  //! The index of the parent node.
  size_t parent;
  //! The base case evaluation.
  double baseCase;

  //! Comparison operator.
  bool operator<(const CoverTreeMapEntry& other) const
  {
    return (score < other.score);
  }
};

}; // namespace tree
}; // namespace mlpack

#endif
