/**
 * @file recursion_sets.hpp
 * @author Ryan Curtin
 *
 * Definition of the RecursionSet object, used during traversal of cover trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_RECURSION_SETS_HPP
#define MLPACK_CORE_TREE_COVER_TREE_RECURSION_SETS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The SingleCoverTreeMapEntry simply holds the information necessary during a
 * single-tree traversal.  The node that is held is the reference node.
 */
template<typename TreeType>
struct SingleCoverTreeMapEntry
{
  // Construct the map entry.
  SingleCoverTreeMapEntry(TreeType* node, double score) :
      node(node), score(score) { /* Nothing to do. */ }

  // Reference node that data is held for.
  TreeType* node;
  // Score of the node (note that this might be the score of the parent, if the
  // base case has not been computed yet).
  double score;

  // Simple comparison for sorting entries.
  inline bool operator<(const SingleCoverTreeMapEntry& other) const
  {
    // For single-tree search, we only need to check the score.
    return (score < other.score);
  }
};
/**
 * The DualCoverTreeMapEntry simply holds the information necessary during a
 * dual-tree traversal.  The node that is held is the reference node.
 */
template<typename TreeType, typename RuleType>
struct DualCoverTreeMapEntry
{
  // Construct the map entry.
  DualCoverTreeMapEntry(TreeType* node,
                        double score,
                        double baseCase,
                        typename RuleType::TraversalInfoType traversalInfo) :
      node(node),
      score(score),
      baseCase(baseCase),
      traversalInfo(std::move(traversalInfo))
  { /* Nothing to do. */ }

  // Reference node that data is held for.
  TreeType* node;
  // Score of the node (note that this might be the score of the parent, if the
  // base case has not been computed yet).
  double score;
  // Base case of the node (note that this might be the base case of the
  // parent).
  double baseCase;
  // Traversal info object from when the last call to Score()/BaseCase() was
  // made with this or an ancestor node combination.
  typename RuleType::TraversalInfoType traversalInfo;

  // Simple comparison for sorting entries.
  inline bool operator<(const DualCoverTreeMapEntry& other) const
  {
    if (score == other.score)
      return (baseCase < other.baseCase);
    else
      return (score < other.score);
  }
};

/**
 * The CoverTreeRecursionSets object holds the information required to recurse a
 * reference tree in a breadth-first manner.  Since the cover tree is made up of
 * levels, this object holds all of the reference nodes that need to be
 * explored, organized into sets for each level.
 *
 * Since the vast majority of reference nodes will simply add children at the
 * next level (or perhaps two levels down), or add leaves, we hold the largest
 * levels as well as the leaf nodes in specific vectors ("hot vectors").  Lower
 * levels are held in a "cold" std::map, which ideally is accessed far less
 * often, or not at all.
 *
 * The structure automatically handles promotion of levels from the cold map to
 * hot vectors, and provides an agnostic interface so that (hopefully) the
 * traversals can use a simple abstraction that sort of resembles a std::map.
 */
template<typename MapEntryType, size_t HotVectorSize = 4>
class CoverTreeRecursionSets
{
 public:
  /**
   * Create an empty set of levels.
   */
  CoverTreeRecursionSets();

  /**
   * Get a std::vector<> containing map entries for a particular scale level.
   * If the scale level does not exist, one will be created and an empty vector
   * will be returned.
   *
   * (This will reuse existing empty vectors if available.)
   */
  inline std::vector<MapEntryType>& GetScaleVector(const int scale);

  // Return whether or not the recursion sets are entirely empty.
  inline bool IsEmpty() const;
  // Return the maximum scale level held in the recursion sets.
  inline int MaxScale() const;

  // Remove all map entries with the given scale, and remove the scale from the
  // hot set if it exists there.
  inline void RemoveScale(const int scale);

  /**
   * This iterator is very limited: it iterates over hot sets, then the leaf
   * vector, then the cold map; but this is not guaranteed to be in scale order!
   * In addition, it does not work if any levels are added or removed during
   * recursion.  It is meant to be used only for read-only iteration.
   */
  class Iterator
  {
   public:
    Iterator(CoverTreeRecursionSets& parent, const bool begin = true) :
        parent(parent),
        hotScaleOrder(arma::sort_index(parent.hotScaleLevels, "descend")),
        hotIndex(0),
        coldMapIter(parent.coldScaleMap.begin()),
        coldMapIterEnd(parent.coldScaleMap.end())
    {
      if (begin)
      {
        // Move to first valid location.
        ++(*this);
      }
    }

    // Determine whether the iterator has completed its iteration.
    inline bool AtEnd() const
    {
      return (hotIndex == HotVectorSize + 2 && coldMapIter == coldMapIterEnd);
    }

    // Increment the iterator.
    inline void operator++()
    {
      // First we iterate over the hot vector.
      if (hotIndex <= HotVectorSize)
      {
        ++hotIndex;
        while (hotIndex < (HotVectorSize + 1) &&
               parent.hotScaleLevels[hotScaleOrder[hotIndex - 1]] == INT_MIN)
          ++hotIndex; // Skip any empty scales.

        // Before settling on the leaf vector, make sure it's not empty.
        if (hotIndex == HotVectorSize + 1 && parent.leafVector.empty())
          ++hotIndex; // Move to the map.
      }
      else if (hotIndex == HotVectorSize + 1)
      {
        // Move from leaf vector to map.
        ++hotIndex;
      }
      else if (coldMapIter != parent.coldScaleMap.end())
      {
        ++coldMapIter;
      }
    }

    // Get the current vector of map entries.
    inline std::vector<MapEntryType>& Vector()
    {
      if (hotIndex <= HotVectorSize)
        return parent.hotScaleVectors[hotScaleOrder[hotIndex - 1]];
      else if (hotIndex == HotVectorSize + 1)
        return parent.leafVector;
      else
        return (*coldMapIter).second;
    }

    // Get the current scale.
    inline int Scale() const
    {
      if (hotIndex <= HotVectorSize)
        return parent.hotScaleLevels[hotScaleOrder[hotIndex - 1]];
      else if (hotIndex == HotVectorSize + 1)
        return INT_MIN;
      else
        return (*coldMapIter).first;
    }

   private:
    // Reference to the parent object (so we can access the level sets).
    CoverTreeRecursionSets& parent;
    // Order of scales in hot set.
    arma::uvec::fixed<HotVectorSize> hotScaleOrder;
    // Current index of 'hot' objects that we are looking at.
    size_t hotIndex;
    // Internally-held iterator for the cold map.
    typename std::map<int, std::vector<MapEntryType>,
        std::greater<int>>::iterator coldMapIter;
    // Internally-held end iterator for the cold map.
    typename std::map<int, std::vector<MapEntryType>,
        std::greater<int>>::iterator coldMapIterEnd;
  };

  // Get an iterator.  Note all the caveats documented for the Iterator class!
  // Iteration should only be performed when no changes are being made to the
  // recursion sets.
  inline Iterator begin() { return Iterator(*this); }

 private:
  // "Hot" scale vectors, which store map entries that we expect to look up
  // frequently.  These are not guaranteed to be ordered, but every scale
  // present in the hot vectors will be greater than any scale present in the
  // cold map.
  std::vector<MapEntryType> hotScaleVectors[HotVectorSize];

  // Scale levels associated with each entry in the hot scale vectors.  A value
  // of INT_MIN means that the hot vector is unallocated to any scale.
  arma::ivec::fixed<HotVectorSize> hotScaleLevels;

  // The cold map stores scale vectors for any scale that cannot be represented
  // in the hot vectors.  The first entry of the map is the scale; all scales in
  // the cold map are less than any scale in the hot vectors.
  std::map<int, std::vector<MapEntryType>, std::greater<int>> coldScaleMap;

  // Specially-held vector of map entries for leaf nodes (where the scale is
  // INT_MIN).
  std::vector<MapEntryType> leafVector;

  // Get the index of the vector to use for the given scale, assuming that scale
  // is *not* `INT_MIN` (in which case the leaf vector should be used).
  //
  // A return value of [0, HotVectorSize) indicates that that element of
  // `hotScaleVectors` represents the map entries for scale `scale`.
  //
  // A return value of HotVectorSize indicates that `coldScaleMap` should be
  // used.
  inline size_t GetScaleIndex(const int scale);
};

} // namespace mlpack

#include "recursion_sets_impl.hpp"

#endif
