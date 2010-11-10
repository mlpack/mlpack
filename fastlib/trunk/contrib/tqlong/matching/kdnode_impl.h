#ifndef KDNODE_IMPL_H
#define KDNODE_IMPL_H

#include "matching.h"

MATCHING_NAMESPACE_BEGIN;

template <typename P, typename N>
    KDNodeStats<P,N>::KDNodeStats(const Matrix &points)
      : KDNode(points), pointStats_(*(new all_point_stats_type(n_points_))), changed_(true)
{
}

template <typename P, typename N>
    KDNodeStats<P,N>::KDNodeStats(KDNodeStats *parent)
      : KDNode(parent), pointStats_(parent->pointStats_), changed_(true)
{
}

template <typename P, typename N>
    KDNodeStats<P,N>::~KDNodeStats()
{
  if (isRoot()) delete &pointStats_;
}

template <typename P, typename N>
    KDNode* KDNodeStats<P,N>::newNode(KDNode *parent)
{
  return new KDNodeStats((KDNodeStats*) parent);
}

template <typename P, typename N>
    void KDNodeStats<P,N>::setPointStats(int index, const point_stats_type &stats)
{
  pointStats_[oldIndex(index)] = stats;
  ((KDNodeStats*)leaf(index))->setChanged(true);
}

template <typename P, typename N>
    void KDNodeStats<P,N>::setChanged(bool changed)
{
  if (changed_ == changed) return;
  changed_ = changed;
  if (changed && !isRoot())
    ((KDNodeStats*)parent())->setChanged(changed);
}

template <typename P, typename N>
    void KDNodeStats<P,N>::visit(bool init)
{
  if (!isChanged()) return;  // the node is unchanged, not neccessary to proceed
  if (isLeaf())
    setLeafStats(init);
  else
  {
    for (unsigned int i = 0; i < children_.size(); i++)
    {
      ((KDNodeStats*) children_[i])->visit(init);
    }
    setNonLeafStats(init);
  }
  setChanged(false);         // the node statistics is refreshed
}

MATCHING_NAMESPACE_END;

#endif // KDNODE_IMPL_H
