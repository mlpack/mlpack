// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file kdtree.h
 *
 * Tools for kd-trees.
 *
 * @experimental
 */

#ifndef TREE_SPKDTREE_H
#define TREE_SPKDTREE_H

#include "spnode.h"
#include "spbounds.h"

#include "base/common.h"
#include "col/arraylist.h"
#include "file/serialize.h"
#include "fx/fx.h"

/* Implementation */

template<typename TPointInfo, typename TNode, typename TParam>
class KdTreeMidpointBuilder {
 public:
  typedef TPointInfo PointInfo;
  typedef TNode Node;
  typedef typename TNode::Bound Bound;
  typedef TParam Param;
  
 private:
  const Param* param_;
  ArrayList<Vector> points_;
  ArrayList<PointInfo> point_info_;
  ArrayList<Node> nodes_;
  ArrayList<index_t> old_from_new_indices_;
  index_t leaf_size_;
  index_t dim_;
  
 public:
  void Init(const Param* param_in_,
      const Matrix& matrix_in, ArrayList<PointInfo> point_info_in) {
    param_ = param_in_;
    
    points_.Init(matrix_in.n_cols());
    for (index_t i = 0; i < points_.size(); i++) {
      Vector v;
      matrix_in.MakeColumnVector(i, &v);
      points_[i].Copy(v);
    }
    
    point_info_.Copy(point_info_in);
    math::MakeIdentityPermutation(matrix_in.n_cols(), &old_from_new_indices_);
    
    nodes_.Init();
    
    dim_ = matrix_in.n_rows();
    
    // TODO: PARAMETERIZE
    leaf_size_ = 5;
  }
  
  void Build();
  
  ArrayList<Vector>& points() { return points_; }
  ArrayList<PointInfo>& point_info() { return point_info_; }
  ArrayList<Node>& nodes() { return nodes_; }
  ArrayList<int>& old_from_new_indices() { return old_from_new_indices_; }

  const ArrayList<Vector>& points() const { return points_; }
  const ArrayList<PointInfo>& point_info() const { return point_info_; }
  const ArrayList<Node>& nodes() const { return nodes_; }
  const ArrayList<int>& old_from_new_indices() const { return old_from_new_indices_; }
  
 private:
  index_t Partition_(
      index_t split_dim, double splitvalue,
      index_t first, index_t count,
      Bound* left_bound, Bound* right_bound);
  void FindBoundingBox_(index_t first, index_t count, Bound *bound);
  void Build_(index_t node_i);
};

template<typename TPointInfo, typename TNode, typename TParam>
void KdTreeMidpointBuilder<TPointInfo, TNode, TParam>::FindBoundingBox_(
    index_t first, index_t count, Bound *bound) {
  index_t end = first + count;
  for (index_t i = first; i < end; i++) {
    *bound |= points_[i];
  }
}
template<typename TPointInfo, typename TNode, typename TParam>
index_t KdTreeMidpointBuilder<TPointInfo, TNode, TParam>::Partition_(
    index_t split_dim, double splitvalue,
    index_t first, index_t count,
    Bound* left_bound, Bound* right_bound) {
  index_t left = first;
  index_t right = first + count - 1;

  /* At any point:
   *
   *   everything < left is correct
   *   everything > right is correct
   */
  for (;;) {
    while (points_[left][split_dim] < splitvalue && likely(left <= right)) {
      *left_bound |= points_[left];
      left++;
    }

    while (points_[right][split_dim] >= splitvalue && likely(left <= right)) {
      *right_bound |= points_[right];
      right--;
    }

    if (unlikely(left > right)) {
      /* left == right + 1 */
      break;
    }

    points_[left].SwapValues(&points_[right]);
    // TODO: If point info has pointers this will incur bad cache performance
    // In the future we rely on OT frozen storage
    mem::Swap(&point_info_[left], &point_info_[right]);

    *left_bound |= points_[left];
    *right_bound |= points_[right];

    index_t t = old_from_new_indices_[left];
    old_from_new_indices_[left] = old_from_new_indices_[right];
    old_from_new_indices_[right] = t;

    DEBUG_ASSERT(left <= right);
    right--;
  }

  DEBUG_ASSERT(left == right + 1);

  return left;
}

template<typename TPointInfo, typename TNode, typename TParam>
void KdTreeMidpointBuilder<TPointInfo, TNode, TParam>::KdTreeMidpointBuilder::Build_(
    index_t node_i) {
  nodes_[node_i].stat().Init(*param_);

  if (nodes_[node_i].count() > leaf_size_) {
    index_t split_dim = BIG_BAD_NUMBER;
    double max_width = -1;

    for (index_t d = 0; d < dim_; d++) {
      double w = nodes_[node_i].bound().get(d).width();

      if (unlikely(w > max_width)) {
        max_width = w;
        split_dim = d;
      }
    }

    double split_val = nodes_[node_i].bound().get(split_dim).mid();

    if (max_width != 0) {
      index_t left_i = nodes_.size();
      index_t right_i = nodes_.size() + 1;
      
      nodes_.AddBack(2);

      nodes_[left_i].bound().Init(dim_);
      nodes_[right_i].bound().Init(dim_);

      index_t split_col = Partition_(split_dim, split_val,
          nodes_[node_i].begin(), nodes_[node_i].count(),
          &nodes_[left_i].bound(),
          &nodes_[right_i].bound());

      DEBUG_MSG(3.0,"split (%d,[%d],%d) split_dim %d on %f (between %f, %f)",
          nodes_[node_i].begin(), split_col,
          nodes_[node_i].begin() + nodes_[node_i].count(), split_dim, split_val,
          nodes_[node_i].bound().get(split_dim).lo,
          nodes_[node_i].bound().get(split_dim).hi);

      nodes_[left_i].Init(nodes_[node_i].begin(),
          split_col - nodes_[node_i].begin());
      nodes_[right_i].Init(split_col,
          nodes_[node_i].begin() + nodes_[node_i].count() - split_col);

      // This should never happen if max_width > 0
      DEBUG_ASSERT(nodes_[left_i].count() != 0 && nodes_[right_i].count() != 0);

      Build_(left_i);
      Build_(right_i);
      
      nodes_[node_i].set_child(0, left_i);
      nodes_[node_i].set_child(1, right_i);
      
      nodes_[node_i].stat().Accumulate(*param_, nodes_[left_i].stat(),
          nodes_[left_i].bound(), nodes_[left_i].count());
      nodes_[node_i].stat().Accumulate(*param_, nodes_[right_i].stat(),
          nodes_[right_i].bound(), nodes_[right_i].count());
      nodes_[node_i].stat().Postprocess(*param_, nodes_[node_i].bound(),
          nodes_[node_i].count());
      
      return;
    }
  }

  nodes_[node_i].set_leaf();
  
  for (index_t i = nodes_[node_i].begin(); i < nodes_[node_i].end(); i++) {
    nodes_[node_i].stat().Accumulate(*param_, points_[i], point_info_[i]);
  }
  nodes_[node_i].stat().Postprocess(*param_, nodes_[node_i].bound(), nodes_[node_i].count());
}

template<typename TPointInfo, typename TNode, typename TParam>
void KdTreeMidpointBuilder<TPointInfo, TNode, TParam>::Build() {
  index_t node_i = 0;

  nodes_.AddBack();
  nodes_[node_i].Init(0, points_.size());
  nodes_[node_i].bound().Init(dim_);
  
  FindBoundingBox_(node_i, points_.size(), &nodes_[node_i].bound());

  Build_(0);
}

#endif
