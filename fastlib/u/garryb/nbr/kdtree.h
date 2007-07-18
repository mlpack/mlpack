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
#include "cachearray.h"

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

 public:
  static void Build(struct datanode *module, const Param &param,
      index_t begin_index, index_t end_index,
      CacheArray<Point> *points_inout, CacheArray<Node> *nodes_create) {
    KdTreeMidpointBuilder builder;
    builder.InitBuild_(module, &param, begin_index, end_index,
        points_inout, nodes_create);
  }

 private:
  const Param* param_;
  CacheArray<Point> points_;
  CacheArray<Node>* nodes_;
  index_t leaf_size_;
  index_t dim_;
  index_t begin_index_;
  index_t end_index_;

 private:
  void InitBuild_(
      struct datanode *module,
      const Param* param_in_,
      index_t begin_index,
      index_t end_index,
      CacheArray<Point> *points_inout,
      CacheArray<Node> *nodes_create) {
    param_ = param_in_;
    begin_index_ = begin_index;
    end_index_ = end_index;

    points_.Init(points_inout, BlockDevice::M_MODIFY);

    nodes_ = nodes_create;

    {
      CacheRead<Point> first_point(&points_, points_.begin_index());
      dim_ = first_point->vec().length();
    }

    leaf_size_ = fx_param_int(module, "leaf_size", 32);
    if (!math::IsPowerTwo(leaf_size_)) {
      NONFATAL("With NBR, it's best to have leaf_size be a power of 2.");
    }

    fx_timer_start(module, "tree_build");
    Build_();
    fx_timer_stop(module, "tree_build");

    points_.Flush(false);
    nodes_->Flush(false);
  }
  index_t Partition_(
      index_t split_dim, double splitvalue,
      index_t begin, index_t count,
      Bound* left_bound, Bound* right_bound);
  void FindBoundingBox_(index_t begin, index_t count, Bound *bound);
  void Build_(index_t node_i);
  void Build_();
};

template<typename TPoint, typename TNode, typename TParam>
void KdTreeMidpointBuilder<TPoint, TNode, TParam>::FindBoundingBox_(
    index_t begin, index_t count, Bound *bound) {
  CacheReadIter<Point> point(&points_, begin);
  index_t end = begin + count;
  for (index_t i = begin; i < end; i++, point.Next()) {
    *bound |= point->vec();
  }
}

template<typename TPoint, typename TNode, typename TParam>
index_t KdTreeMidpointBuilder<TPoint, TNode, TParam>::Partition_(
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
    for (;;) {
      if (unlikely(left_i > right_i)) return left_i;
      CacheRead<Point> left_v(&points_, left_i);
      if (left_v->vec().get(split_dim) >= splitvalue) {
        *right_bound |= left_v->vec();
        break;
      }
      *left_bound |= left_v->vec();
      left_i++;
    }

    for (;;) {
      if (unlikely(left_i > right_i)) return left_i;
      CacheRead<Point> right_v(&points_, right_i);
      if (right_v->vec().get(split_dim) < splitvalue) {
        *left_bound |= right_v->vec();
        break;
      }
      *right_bound |= right_v->vec();
      right_i--;
    }

    points_.Swap(left_i, right_i);

    DEBUG_ASSERT(left_i <= right_i);
    right_i--;
  }

  abort();
}

template<typename TPoint, typename TNode, typename TParam>
void KdTreeMidpointBuilder<TPoint, TNode, TParam>::Build_(
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

    if (max_width != 0) {
      index_t left_i = nodes_->Alloc();
      index_t right_i = nodes_->Alloc();
      Node *left = nodes_->StartWrite(left_i);
      Node *right = nodes_->StartWrite(right_i);

      left->bound().Reset();
      right->bound().Reset();

      index_t split_col;
      index_t begin_col = node->begin();
      index_t end_col = node->end();
      // attempt to make all leaves of identical size
      index_t goal_col = (begin_col + end_col + leaf_size_) / leaf_size_ / 2 * leaf_size_;
      double split_val;
      SpRange current_range = node->bound().get(split_dim);
      typename Node::Bound left_bound;
      typename Node::Bound right_bound;
      left_bound.Init(dim_);
      right_bound.Init(dim_);

      for (;;) {
        split_val = current_range.interpolate(
            (goal_col - begin_col) / double(end_col - begin_col));
        //fprintf(stderr, "(%d..(%d..%d..%d)..%d) (%f..%f) %f\n",
        //    node->begin(), begin_col, goal_col, end_col, node->end(),
        //    current_range.lo, current_range.hi, split_val);

        left_bound.Reset();
        right_bound.Reset();
        split_col = Partition_(split_dim, split_val,
              begin_col, end_col - begin_col,
              &left_bound, &right_bound);

        // To do midpoint-based tree-building, make the following branch
        // always happen.
        if (split_col == goal_col) {
          left->bound() |= left_bound;
          right->bound() |= right_bound;
          break;
        } else if (split_col < goal_col) {
          left->bound() |= left_bound;
          current_range = right_bound.get(split_dim);
          if (current_range.width() == 0) {
            right->bound() |= right_bound;
            break;
          }
          begin_col = split_col;
        } else if (split_col > goal_col) {
          right->bound() |= right_bound;
          current_range = left_bound.get(split_dim);
          if (current_range.width() == 0) {
            left->bound() |= left_bound;
            break;
          }
          end_col = split_col;
        }
      }

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
      left->stat().Postprocess(*param_, node->bound(),
          node->count());
      node->stat().Accumulate(*param_, right->stat(),
          right->bound(), right->count());
      right->stat().Postprocess(*param_, node->bound(),
          node->count());

      leaf = false;
      nodes_->StopWrite(left_i);
      nodes_->StopWrite(right_i);
    } else {
      NONFATAL("There is probably a bug somewhere else - %"LI"d points are all identical.",
              node->count());
    }
  }

  if (leaf) {
    node->set_leaf();

    for (index_t i = node->begin(); i < node->end(); i++) {
      CacheRead<Point> point(&points_, i);
      node->stat().Accumulate(*param_, *point);
    }

    //fprintf(stderr, "leaf %d..%d (%d)\n", node->begin(), node->end(), node->count());
  }

  nodes_->StopWrite(node_i);
}

template<typename TPoint, typename TNode, typename TParam>
void KdTreeMidpointBuilder<TPoint, TNode, TParam>::Build_() {
  index_t node_i = nodes_->Alloc();
  Node *node = nodes_->StartWrite(node_i);

  DEBUG_SAME_INT(node_i, 0);

  node->set_range(begin_index_, end_index_);

  FindBoundingBox_(node->begin(), node->end(), &node->bound());

  Build_(node_i);
  node->stat().Postprocess(*param_, node->bound(), node->count());

  nodes_->StopWrite(node_i);
}

#endif
