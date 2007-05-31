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
#include "cache.h"

#include "base/common.h"
#include "col/arraylist.h"
#include "file/serialize.h"
#include "fx/fx.h"

/* Implementation */

template<typename TPoint, typename TNode, typename TParam>
class KdTreeMidpointBuilder {
 public:
  typedef TNode Node;
  typedef TPoint Point;
  typedef typename TNode::Bound Bound;
  typedef TParam Param;

 private:
  const Param* param_;
  CacheArray<Point> points_;
  CacheArray<Node>* nodes_;
  index_t leaf_size_;
  index_t dim_;

 public:
  void InitBuild(
      struct datanode *module,
      const Param* param_in_,
      CacheArray<Point> *points_inout,
      CacheArray<Node> *nodes_out) {
    param_ = param_in_;

    points_.Init(points_inout, BlockDevice::MODIFY);
    nodes_ = nodes_out;

    CacheRead<Point> first_point(&points_, points_.begin_index());
    dim_ = first_point->vec().length();

    leaf_size_ = fx_param_int(module, "leaf_size", 20);

    Build_();
    
    points_.Flush();
    nodes_->Flush();
  }
  
 private:
  index_t Partition_(
      index_t split_dim, double splitvalue,
      index_t first, index_t count,
      Bound* left_bound, Bound* right_bound);
  void FindBoundingBox_(index_t first, index_t count, Bound *bound);
  void Build_(index_t node_i);
  void Build_();
};

template<typename TNode, typename TParam>
void KdTreeMidpointBuilder<TNode, TParam>::FindBoundingBox_(
    index_t first, index_t count, Bound *bound) {
  CacheReadIterator<Point> point(&points_, i);
  index_t end = first + count;
  for (index_t i = first; i < end; i++, point.Next()) {
    *bound |= point->vec();
  }
}
template<typename TNode, typename TParam>
index_t KdTreeMidpointBuilder<TNode, TParam>::Partition_(
    index_t split_dim, double splitvalue,
    index_t begin, index_t count,
    Bound* left_bound, Bound* right_bound) {
  index_t left_i = begin;
  index_t right_i = begin + count - 1;

  /* At any point:
   *
   *   everything < left_i is correct
   *   everything > right_i is correct
   */
  for (;;) {
    while (1) {
      CacheRead<Point> left_v(&points_, left_i);
      if (left_v->vec().get(split_dim) >= splitvalue
          || unlikely(left_i > right_i)) {
        *right_bound |= left_v->vec();
        break;
      }
      *left_bound |= left_v->vec();
      left_i++;
    }

    while (1) {
      CacheRead<Point> right_v(&points, right_i);
      if (right_v->get(split_dim) < splitvalue
          || unlikely(left_i > right_i)) {
        *left_bound |= right_v->vec();
        break;
      }
      *right_bound |= right_v->vec();
      right_i--;
    }

    if (unlikely(left_i > right_i)) {
      break;
    }

    points_.Swap(left_i, right_i);

    DEBUG_ASSERT(left_i <= right_i);
    right_i--;
  }

  DEBUG_ASSERT(left_i == right_i + 1);

  return left_i;
}

template<typename TNode, typename TParam>
void KdTreeMidpointBuilder<TNode, TParam>::KdTreeMidpointBuilder::Build_(
    index_t node_i) {
  Node *node = nodes_->StartWrite(node_i);
  bool leaf = true;
  
  if (node->count() > leaf_size_) {
    index_t split_dim = BIG_BAD_NUMBER;
    double max_width = -1;

    // Short loop to find widest dimension
    for (index_t d = 0; d < dim_; d++) {
      double w = node->bound().get(d).width();

      if (unlikely(w > max_width)) {
        max_width = w;
        split_dim = d;
      }
    }

    
    DEBUG_ASSERT_MSG(max_width != 0,
        "There is probably a bug somewhere else - your points are all identical.");

    if (max_width != 0) {
      index_t left_i = nodes_->Alloc();
      index_t right_i = nodes_->Alloc();
      Node *left = nodes_->StartWrite(left_i);
      Node *right = nodes_->StartWrite(right_i);

      index_t split_col;
      double split_val = node->bound().get(split_dim).mid();
      
      left->bound().Reset();
      right->bound().Reset();
      split_col = Partition_(split_dim, split_val,
            node->begin(), node->count(),
            &left->bound(),
            &right->bound());
      
      DEBUG_MSG(3.0,"split (%d,[%d],%d) split_dim %d on %f (between %f, %f)",
          node->begin(), split_col,
          node->begin() + node->count(), split_dim, split_val,
          node->bound().get(split_dim).lo,
          node->bound().get(split_dim).hi);

      left->set_range(node->begin(), split_col - node->begin());
      right->set_range(split_col, node->end() - split_col);

      // This should never happen if max_width > 0
      DEBUG_ASSERT(left->count() != 0 && right->count() != 0);

      Build_(left_i);
      Build_(right_i);
      
      node->set_child(0, left_i);
      node->set_child(1, right_i);
      
      node->stat().Accumulate(*param_, left->stat(),
          left->bound(), left->count());
      node->stat().Accumulate(*param_, right->stat(),
          right->bound(), right->count());
      node->stat().Postprocess(*param_, node->bound(),
          node->count());
      
      leaf = false;
      nodes_->StopWrite(left_i);
      nodes_->StopWrite(right_i);
    }
  }

  if (leaf) {
    node->set_leaf();
  
    for (index_t i = node->begin(); i < node->end(); i++) {
      CacheRead<Point> point(&points_, i);
      node->stat().Accumulate(*param_, *point);
    }
    node->stat().Postprocess(*param_, node->bound(), node->count());
  }
  
  nodes_->StopWrite(node_i);
}

template<typename TNode, typename TParam>
void KdTreeMidpointBuilder<TNode, TParam>::Build_() {
  index_t node_i = nodes_->Alloc();
  Node *node = nodes_->StartWrite(node_i);
  
  DEBUG_SAME_INT(node_i, 0);

  node->set_range(points_.begin_index(), points_.end_index());
  
  FindBoundingBox_(node->begin(), node->end(), &node->bound());

  Build_(node_i);
  
  nodes_->StopWrite(node_i);
}

#endif
