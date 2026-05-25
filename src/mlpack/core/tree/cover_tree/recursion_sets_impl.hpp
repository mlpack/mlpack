/**
 * @file recursion_sets_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the CoverTreeRecursionSet object, used during traversal of
 * cover trees.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_RECURSION_SETS_IMPL_HPP
#define MLPACK_CORE_TREE_COVER_TREE_RECURSION_SETS_IMPL_HPP

#include "recursion_sets.hpp"

namespace mlpack {

template<typename TreeType, typename RuleType, size_t HotVectorSize>
inline CoverTreeRecursionSets<TreeType, RuleType, HotVectorSize>::
CoverTreeRecursionSets()
{
  hotScaleLevels.fill(INT_MIN);
}

template<typename TreeType, typename RuleType, size_t HotVectorSize>
inline std::vector<CoverTreeMapEntry<TreeType, RuleType>>&
CoverTreeRecursionSets<TreeType, RuleType, HotVectorSize>::GetScaleVector(
    const int scale)
{
  if (scale == INT_MIN)
    return leafVector;

  const size_t index = GetScaleIndex(scale);
  if (index < HotVectorSize)
    return hotScaleVectors[index];
  else
    return coldScaleMap[scale];
}

template<typename TreeType, typename RuleType, size_t HotVectorSize>
inline bool CoverTreeRecursionSets<TreeType, RuleType, HotVectorSize>::IsEmpty()
    const
{
  if (coldScaleMap.size() != 0)
    return false;

  if (leafVector.size() != 0)
    return false;

  for (size_t i = 0; i < HotVectorSize; ++i)
    if (hotScaleLevels[i] != INT_MIN)
      return false;

  return true;
}

template<typename TreeType, typename RuleType, size_t HotVectorSize>
inline int CoverTreeRecursionSets<TreeType, RuleType, HotVectorSize>::MaxScale()
    const
{
  int maxScale = INT_MIN;
  for (size_t i = 0; i < HotVectorSize; ++i)
  {
    if (hotScaleLevels[i] > maxScale)
      maxScale = hotScaleLevels[i];
  }

  // No need to check the cold map, since anything in the hot vectors will have
  // higher scale, and the cold map is emptied before the hot vectors.

  return maxScale;
}

template<typename TreeType, typename RuleType, size_t HotVectorSize>
inline void CoverTreeRecursionSets<TreeType, RuleType, HotVectorSize>::
RemoveScale(const int scale)
{
  if (scale == INT_MIN)
  {
    leafVector.clear();
    return;
  }

  // Try to find the scale in the hot sets.
  for (size_t i = 0; i < HotVectorSize; ++i)
  {
    if (hotScaleLevels[i] == scale)
    {
      // Pull something from the cold map, if needed.
      if (coldScaleMap.size() > 0)
      {
        hotScaleLevels[i] = coldScaleMap.begin()->first;
        hotScaleVectors[i] = std::move(coldScaleMap.begin()->second);
      }
      else
      {
        hotScaleLevels[i] = INT_MIN;
        hotScaleVectors[i].clear();
      }
      return;
    }
  }
}

template<typename TreeType, typename RuleType, size_t HotVectorSize>
inline void CoverTreeRecursionSets<TreeType, RuleType, HotVectorSize>::Clear()
{
  for (size_t i = 0; i < HotVectorSize; ++i)
  {
    hotScaleLevels[i] = INT_MIN;
    hotScaleVectors[i].clear();
  }
  coldScaleMap.clear();
  leafVector.clear();
}

template<typename TreeType, typename RuleType, size_t HotVectorSize>
inline size_t CoverTreeRecursionSets<TreeType, RuleType, HotVectorSize>::
GetScaleIndex(const int scale)
{
  // Loop through the hot scales.  Allocate one, if it's unallocated.
  int minScale = INT_MAX;
  int firstMinScale = HotVectorSize;
  for (size_t i = 0; i < HotVectorSize; ++i)
  {
    if (hotScaleLevels[i] == INT_MIN)
      firstMinScale = i; // We will allocate this vector.
    if (hotScaleLevels[i] == scale)
      return i;
    if (hotScaleLevels[i] < minScale)
      minScale = hotScaleLevels[i];
  }
  if (firstMinScale != HotVectorSize)
  {
    hotScaleLevels[firstMinScale] = scale;
    return firstMinScale;
  }

  if (scale > minScale)
  {
    // In this case, we are encountering a scale that *should* be in the hot
    // scale set but isn't!  So, we need to evict an existing one.
    size_t minScaleIndex = HotVectorSize;
    minScale = INT_MAX;
    for (size_t i = 0; i < HotVectorSize; ++i)
    {
      if (hotScaleLevels[i] < minScale)
      {
        minScaleIndex = i;
        minScale = hotScaleLevels[i];
      }
    }

    // Move the smallest-scale vector into the cold map and claim the hot
    // vector for this scale.
    coldScaleMap[minScale] = std::move(hotScaleVectors[minScaleIndex]);
    hotScaleLevels[minScaleIndex] = scale;
    return minScaleIndex;
  }

  // If there are no hot scales for this scale level, we will need to put it in
  // the std::map.
  return HotVectorSize;
}

} // namespace mlpack

#endif
