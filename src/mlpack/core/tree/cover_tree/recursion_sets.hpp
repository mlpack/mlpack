/**
 * @file recursion_sets.hpp
 * @author Ryan Curtin
 *
 * Definition of the RecursionSet object, used during traversal of cover trees.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_RECURSION_SETS_HPP
#define MLPACK_CORE_TREE_COVER_TREE_RECURSION_SETS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename TreeType, typename RuleType>
struct CoverTreeMapEntry
{
  TreeType* node;
  double score;
  double baseCase;
  typename RuleType::TraversalInfoType traversalInfo;
  inline bool operator<(const CoverTreeMapEntry& other) const
  {
    if (score == other.score)
      return (baseCase < other.baseCase);
    else
      return (score < other.score);
  }
};

template<typename TreeType, typename RuleType, size_t HotVectorSize = 4>
class CoverTreeRecursionSets
{
 public:
  CoverTreeRecursionSets();

  inline std::vector<CoverTreeMapEntry<TreeType, RuleType>>& GetScaleVector(
      const int scale);

  inline bool IsEmpty() const;
  inline int MaxScale() const;

  inline void RemoveScale(const int scale);
  inline void Clear();

  class Iterator
  {
   public:
    Iterator(CoverTreeRecursionSets& parent, const bool begin = true) :
        parent(parent),
        hotIndex(begin ? 0 : HotVectorSize + 2 /* past the leaf vector */),
        coldMapIter(begin ? parent.coldScaleMap.begin() :
            parent.coldScaleMap.end())
    {
      if (begin)
      {
        // Move to first valid location.
        ++(*this);
      }
    }

    inline bool operator!=(const Iterator& other) const
    {
      if (&parent != &other.parent)
        return true;
      if (hotIndex != other.hotIndex)
        return true;
      if (coldMapIter != other.coldMapIter)
        return true;

      return false;
    }

    inline void operator++()
    {
      // First we iterate over the hot vector.
      if (hotIndex <= HotVectorSize)
      {
        ++hotIndex;
        while (hotIndex < (HotVectorSize + 1) &&
               parent.hotScaleLevels[hotIndex - 1] == INT_MIN)
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

    inline std::vector<CoverTreeMapEntry<TreeType, RuleType>>& Vector()
    {
      if (hotIndex <= HotVectorSize)
        return parent.hotScaleVectors[hotIndex - 1];
      else if (hotIndex == HotVectorSize + 1)
        return parent.leafVector;
      else
        return (*coldMapIter).second;
    }

    inline int Scale() const
    {
      if (hotIndex <= HotVectorSize)
        return parent.hotScaleLevels[hotIndex - 1];
      else if (hotIndex == HotVectorSize + 1)
        return INT_MIN;
      else
        return (*coldMapIter).first;
    }

   private:
    CoverTreeRecursionSets& parent;
    size_t hotIndex;
    typename std::map<int, std::vector<CoverTreeMapEntry<TreeType, RuleType>>,
        std::greater<int>>::iterator coldMapIter;
  };

  inline Iterator begin() { return Iterator(*this); }
  inline Iterator end() { return Iterator(*this, false); }

 private:
  std::vector<CoverTreeMapEntry<TreeType, RuleType>> hotScaleVectors[HotVectorSize];
  arma::ivec::fixed<HotVectorSize> hotScaleLevels;
  std::map<int, std::vector<CoverTreeMapEntry<TreeType, RuleType>>, std::greater<int>> coldScaleMap;
  std::vector<CoverTreeMapEntry<TreeType, RuleType>> leafVector;

  inline size_t GetScaleIndex(const int scale);
};

} // namespace mlpack

#include "recursion_sets_impl.hpp"

#endif
