#ifndef KDNODE_H
#define KDNODE_H

#include <fastlib/fastlib.h>
#include <vector>
#include "matching.h"

MATCHING_NAMESPACE_BEGIN;

class KDNode
{
protected:
  typedef std::vector<int>          index_type;
  typedef std::vector<KDNode*>      node_list_type;
  class SplitInfo
  {
  public:
    int                   splitDim_;
    double                splitVal_;
  };

  const Matrix&           points_;
  int                     n_dim_, n_points_, dfsIndex_;
  index_type              &oldIndex_;
  node_list_type          &pointToLeaf_;
  SplitInfo               split_;
  node_list_type          children_;
  KDNode                  *parent_;

  void splitOnDim(int dim);
  double selectMedian(int dim) const;
  /** try split the current node using split information in split_ */
  bool splitNode();
  virtual KDNode* newNode(KDNode* parent);

  KDNode(KDNode* parent);
public:
  KDNode(const Matrix& points);
  virtual ~KDNode();
  void split(int minSize = 10);

  int n_points() const;
  int n_dim() const;
  int oldIndex(int index) const;
  void getPoint(int index, Vector& point) const;
  double get(int dim, int index) const;
  bool isRoot() const;
  bool isLeaf() const;
  int n_children() const;
  KDNode* child(int index) const;
  KDNode* parent() const;
  KDNode* leaf(int index) const;

  std::string toString(int depth = 0) const;
};

template <typename P, typename N>
    class KDNodeStats : public KDNode
{
protected:
  typedef P                             point_stats_type;
  typedef N                             node_stats_type;
  typedef std::vector<P>                all_point_stats_type;

  all_point_stats_type    &pointStats_;
  node_stats_type         nodeStats_;
  bool                    changed_;

  virtual KDNode* newNode(KDNode* parent);

  KDNodeStats(KDNodeStats* parent);
public:
  KDNodeStats(const Matrix& points);
  virtual ~KDNodeStats();

  const point_stats_type& pointStats(int index) const { return pointStats_.at(oldIndex(index)); }
  const node_stats_type& nodeStats() const { return nodeStats_; }
  void setPointStats(int index, const point_stats_type& stats);
  bool isChanged() const { return changed_; }
  void setChanged(bool changed);

  /** Depth first search to visit all nodes in the subtree and refresh statistics
    * init = true : initialization phase
    */
  void visit(bool init = false);

  /** Implement the following 2 methods to calculate node statistics */
  void setLeafStats(bool init);
  void setNonLeafStats(bool init);

  /** Implement following method to convert this object to string */
  std::string toString(int depth = 0) const;
};

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

#endif // KDNODE_H
