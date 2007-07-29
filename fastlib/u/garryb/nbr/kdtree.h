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

/**
 * Single-threaded kd-tree builder.
 *
 * Rearranges points in place and attempts to take advantage of the block
 * structure.
 *
 * The algorithm uses a combination of midpoint and median splits.
 * At the higher levels of the tree, a median-like split is done such that
 * the split falls on the block boundary (or otherwise specified chunk_size)
 * that is closest to the middle index.  Once the number of points
 * considered is smaller than the chunk size, midpoint splits are done.
 * The median splits simplify load balancing and allow more efficient
 * storage of data, and actually help the dual-tree algorithm in the
 * initial few layers -- however, the midpoint splits help to separate
 * outliers from the rest of the data.  Leaves are created once the number
 * of points is at most leaf_size.
 */
template<typename TPoint, typename TNode, typename TParam>
class KdTreeHybridBuilder {
 public:
  typedef TNode Node;
  typedef TPoint Point;
  typedef typename TNode::Bound Bound;
  typedef TParam Param;

 public:
  /**
   * Builds a kd-tree.
   *
   * See class comments.
   *
   * @param module module for tuning parameters: leaf_size (maximum
   *        number of points per leaf), and chunk_size (rounding granularity
   *        for median splits)
   * @param param parameters needed by the bound or other structures
   * @param begin_index the first index that I'm building
   * @param end_index one beyond the last index
   * @param points_inout the points, to be reordered
   * @param nodes_create the nodes, which will be allocated one by one
   */
  static void Build(struct datanode *module, const Param &param,
      index_t begin_index, index_t end_index,
      DistributedCache *points_inout, DistributedCache *nodes_create) {
    KdTreeHybridBuilder builder;
    builder.Doit(module, &param, begin_index, end_index,
        points_inout, nodes_create);
  }

 private:
  const Param* param_;
  CacheArray<Point> points_;
  CacheArray<Node> nodes_;
  index_t leaf_size_;
  index_t chunk_size_;
  index_t dim_;
  index_t begin_index_;
  index_t end_index_;
  index_t n_points_;

 public:
  void Doit(
      struct datanode *module,
      const Param* param_in_,
      index_t begin_index,
      index_t end_index,
      DistributedCache *points_inout,
      DistributedCache *nodes_create) {
    param_ = param_in_;
    begin_index_ = begin_index;
    end_index_ = end_index;
    n_points_ = end_index_ - begin_index_;

    points_.Init(points_inout, BlockDevice::M_MODIFY);
    nodes_.Init(nodes_create, BlockDevice::M_CREATE);

    {
      CacheRead<Point> first_point(&points_, points_.begin_index());
      dim_ = first_point->vec().length();
    }

    leaf_size_ = fx_param_int(module, "leaf_size", 32);
    // NOTE: Chunk size is now required to be same as n_block_elems, so that
    // each block of points corresponds to exactly one node in the tree.
    chunk_size_ = points_.n_block_elems();

    fx_timer_start(module, "tree_build");
    Build_();
    fx_timer_stop(module, "tree_build");
  }

  index_t Partition_(
      index_t split_dim, double splitvalue,
      index_t begin, index_t count,
      Bound* left_bound, Bound* right_bound);
  void FindBoundingBox_(index_t begin, index_t count, Bound *bound);
  /**
   * Builds the tree, assigning this portion of the tree to a subset of
   * machines.
   */
  void Build_(index_t node_i, int begin_rank, int end_rank);
  void Build_();
};

template<typename TPoint, typename TNode, typename TParam>
void KdTreeHybridBuilder<TPoint, TNode, TParam>::FindBoundingBox_(
    index_t begin, index_t count, Bound *bound) {
  CacheReadIter<Point> point(&points_, begin);
  for (index_t i = count; i--; point.Next()) {
    *bound |= point->vec();
  }
}

template<typename TPoint, typename TNode, typename TParam>
index_t KdTreeHybridBuilder<TPoint, TNode, TParam>::Partition_(
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
void KdTreeHybridBuilder<TPoint, TNode, TParam>::Build_(
    index_t node_i, int begin_rank, int end_rank) {
  Node *node = nodes_.StartWrite(node_i);
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
      // Let's try to divide the machines in half at this point.
      int split_rank = (begin_rank + end_rank) / 2;
      typename Node::Bound final_left_bound;
      typename Node::Bound final_right_bound;

      final_left_bound.Init(dim_);
      final_right_bound.Init(dim_);

      index_t split_col;
      index_t begin_col = node->begin();
      index_t end_col = node->end();
      // attempt to make all leaves of identical size
      double split_val;
      SpRange current_range = node->bound().get(split_dim);

      if (index_t(node->count()) == index_t(points_.n_block_elems())) {
        // We got one block of points!  Let's give away ownership.
        points_.cache()->GiveOwnership(
            points_.Blockid(node->begin()),
            begin_rank);
      }

      if (node->count() <= chunk_size_) {
        // perform a midpoint split
        split_val = current_range.mid();
        split_col = Partition_(split_dim, split_val,
              begin_col, end_col - begin_col,
              &final_left_bound, &final_right_bound);
      } else {
        index_t goal_col;
        typename Node::Bound left_bound;
        typename Node::Bound right_bound;
        left_bound.Init(dim_);
        right_bound.Init(dim_);

        if (likely(begin_rank - end_rank <= 1)) {
          // All points will go on the same machine, so do median split.
          goal_col = (begin_col + end_col) / 2;
        } else {
          // We're distributing these between machines.  Let's make sure
          // we give roughly even work to the machines.  What we do is
          // pretend the points are distributed as equally as possible, by
          // using the global number of machines and points, to avoid errors
          // interoduced by doing this split computation recursively.
          goal_col = (uint64(split_rank) * 2 * n_points_ + rpc::n_peers())
              / rpc::n_peers() / 2;
        }

        // Round the goal to the nearest block.
        goal_col = (goal_col + chunk_size_ / 2) / chunk_size_ * chunk_size_;

        for (;;) {
          // use linear interpolation to guess the value to split on.
          // this to lead to convergence rather quickly.
          split_val = current_range.interpolate(
              (goal_col - begin_col) / double(end_col - begin_col));

          left_bound.Reset();
          right_bound.Reset();
          split_col = Partition_(split_dim, split_val,
                begin_col, end_col - begin_col,
                &left_bound, &right_bound);

          if (split_col == goal_col) {
            final_left_bound |= left_bound;
            final_right_bound |= right_bound;
            break;
          } else if (split_col < goal_col) {
            final_left_bound |= left_bound;
            current_range = right_bound.get(split_dim);
            if (current_range.width() == 0) {
              // right_bound straddles the boundary, force it to break up
              final_right_bound |= right_bound;
              final_left_bound |= right_bound;
              split_col = goal_col;
              break;
            }
            begin_col = split_col;
          } else if (split_col > goal_col) {
            final_right_bound |= right_bound;
            current_range = left_bound.get(split_dim);
            if (current_range.width() == 0) {
              // left_bound straddles the boundary, force it to break up
              final_left_bound |= left_bound;
              final_right_bound |= left_bound;
              split_col = goal_col;
              break;
            }
            end_col = split_col;
          }
        }

        DEBUG_ASSERT(split_col % points_.n_block_elems() == 0);
      }

      DEBUG_MSG(3.0,"split (%d,[%d],%d) split_dim %d on %f (between %f, %f)"    node->begin(), split_col,
          node->begin() + node->count(), split_dim, split_val,
          node->bound().get(split_dim).lo,
          node->bound().get(split_dim).hi);

      index_t left_i = nodes_.AllocD(begin_rank);
      Node *left = nodes_.StartWrite(left_i);
      left->bound().Reset();
      left->bound() |= final_left_bound;
      left->set_range(node->begin(), split_col - node->begin());
      Build_(left_i, begin_rank, split_rank);
      left->stat().Postprocess(*param_, node->bound(),
          node->count());
      node->stat().Accumulate(*param_, left->stat(),
          left->bound(), left->count());
      nodes_.StopWrite(left_i);

      index_t right_i = nodes_.AllocD(split_rank);
      Node *right = nodes_.StartWrite(right_i);
      right->bound().Reset();
      right->bound() |= final_right_bound;
      right->set_range(split_col, node->end() - split_col);
      Build_(right_i, split_rank, end_rank);
      right->stat().Postprocess(*param_, node->bound(),
          node->count());
      node->stat().Accumulate(*param_, right->stat(),
          right->bound(), right->count());
      nodes_.StopWrite(right_i);

      node->set_child(0, left_i);
      node->set_child(1, right_i);

      leaf = false;
    } else {
      NONFATAL("There is probably a bug somewhere else - "
          "%"LI"d points are all identical.",
          node->count());
    }
  }

  if (leaf) {
    node->set_leaf();
    // ensure leaves don't straddle block boundaries
    DEBUG_SAME_INT(node->begin() / points_.n_block_elems(),
        (node->end() - 1) / points_.n_block_elems());
    for (index_t i = node->begin(); i < node->end(); i++) {
      CacheRead<Point> point(&points_, i);
      node->stat().Accumulate(*param_, *point);
    }
  }

  nodes_.StopWrite(node_i);
}

template<typename TPoint, typename TNode, typename TParam>
void KdTreeHybridBuilder<TPoint, TNode, TParam>::Build_() {
  int begin_rank = 0;
  int end_rank = rpc::n_peers();
  index_t node_i = nodes_.AllocD(begin_rank);
  Node *node = nodes_.StartWrite(node_i);

  DEBUG_SAME_INT(node_i, 0);

  node->set_range(begin_index_, end_index_);

  FindBoundingBox_(node->begin(), node->end(), &node->bound());

  Build_(node_i, begin_rank, end_rank);
  node->stat().Postprocess(*param_, node->bound(), node->count());

  nodes_.StopWrite(node_i);
}

template<typename TParam, typename TPoint, typename TNode>
class SpKdTree {
 public:
  typedef TParam Param;
  typedef TPoint Point;
  typedef TNode Node;

 private:
  struct Config {
    Param *param;
    int nodes_block;
    int points_block;
    int n_points;
    int dim;

    OT_DEF(Config) {
      OT_PTR(param);
      OT_MY_OBJECT(param);
      OT_MY_OBJECT(nodes_block);
      OT_MY_OBJECT(points_block);
      OT_MY_OBJECT(n_points);
      OT_MY_OBJECT(dim);
    }
  };
  
  enum { MEGABYTE = 1048576 };

 private:
  DistributedCache points_;
  DistributedCache nodes_;
  datanode *points_module_;
  datanode *nodes_module_;
  int points_channel_;
  int nodes_channel_;
  int nodes_mb_;
  int points_mb_;
  Config *config_;

 public:
  SpKdTree() {}
  ~SpKdTree() {
    delete config_;
  }

  /**
   * Loads a tree and initializes.
   *
   * @param parampp a double-pointer to a heap-allocated param object.
   *        this will detelete *parampp and set *parampp to an object
   *        allocated with 'new'
   * @param param_tag this tag is sent to param when doing the initialization
   * @param base_channel the base channel number -- must give use a region
   *        of 10 free channels, so if base_channel is 400, then 400 to 409
   *        will be used
   */
  void Init(Param **parampp, int param_tag,
      int base_channel, datanode *module);

  DistributedCache& points() { return points_; }
  DistributedCache& nodes() { return nodes_; }
  
  index_t nodes_block() const { return config_->nodes_block; }
  index_t points_block() const { return config_->points_block; }
  index_t n_points() const { return config_->n_points; }

 private:
  void MasterLoadData_(
      Config *config, Param *param, int tag, datanode *module);
  void MasterBuildTree_(
      Config *config, Param *param, int tag, datanode *module);
};

template<typename TParam, typename TPoint, typename TNode>
void SpKdTree<TParam, TPoint, TNode>::Init(Param **parampp, int param_tag,
    int base_channel, datanode *module) {
  points_module_ = fx_submodule(module, "points", "points");
  nodes_module_ = fx_submodule(module, "nodes", "nodes");
  points_channel_ = base_channel + 0;
  nodes_channel_ = base_channel + 1;
  points_mb_ = fx_param_int(points_module_, "mb", 400);
  nodes_mb_ = fx_param_int(nodes_module_, "mb", 100);
  Broadcaster<Config> broadcaster;
  if (rpc::is_root()) {
    Config config;
    MasterLoadData_(&config, *parampp, param_tag, module);
    MasterBuildTree_(&config, *parampp, param_tag, module);
    config.param = *parampp;
    broadcaster.SetData(config);
  } else {
    CacheArray<Point>::InitDistributedCacheWorker(points_channel_,
       size_t(points_mb_) * MEGABYTE, &points_);
    CacheArray<Node>::InitDistributedCacheWorker(nodes_channel_,
       size_t(nodes_mb_) * MEGABYTE, &nodes_);
  }
  points_.StartSync();
  nodes_.StartSync();
  points_.WaitSync();
  nodes_.WaitSync();
  broadcaster.Doit(base_channel + 2);
  config_ = new Config(broadcaster.get());
  delete *parampp;
  *parampp = new Param(*config_->param);
}

template<typename TParam, typename TPoint, typename TNode>
void SpKdTree<TParam, TPoint, TNode>::MasterLoadData_(
    Config *config, Param *param, int tag, datanode *module) {
  config->points_block = fx_param_int(points_module_, "block", 1024);

  fprintf(stderr, "master: Reading data\n");

  fx_timer_start(module, "read");
  TextLineReader reader;
  if (FAILED(reader.Open(fx_param_str_req(module, "")))) {
    FATAL("Could not open data file '%s'", fx_param_str_req(module, ""));
  }
  DatasetInfo schema;
  schema.InitFromFile(&reader, "data");
  config->dim = schema.n_features();

  TPoint default_point;
  default_point.vec().Init(config->dim);
  default_point.vec().SetZero();
  param->InitPointExtras(tag, &default_point);

  CacheArray<Point>::InitDistributedCacheMaster(
      points_channel_, config->points_block,
      size_t(points_mb_) * MEGABYTE, default_point,
      &points_);
  CacheArray<Point> points_array;
  points_array.Init(&points_, BlockDevice::M_CREATE);
  index_t i = 0;

  for (;;) {
    i = points_array.AllocD(rpc::rank(), 1);
    CacheWrite<Point> point(&points_array, i);
    bool is_done;
    success_t rv = schema.ReadPoint(&reader, point->vec().ptr(), &is_done);
    if (unlikely(FAILED(rv))) {
      FATAL("Data file has problems");
    }
    if (is_done) {
      break;
    }
  }

  config->n_points = i;
  param->Bootstrap(tag, config->dim, config->n_points);

  fx_timer_stop(module, "read");
}

template<typename TParam, typename TPoint, typename TNode>
void SpKdTree<TParam, TPoint, TNode>::MasterBuildTree_(
    Config *config, Param *param, int tag, datanode *module) {
  config->nodes_block = fx_param_int(nodes_module_, "block", 512);

  fprintf(stderr, "master: Building tree\n");
  fx_timer_start(module, "tree");
  Node example_node;
  example_node.Init(config->dim, *param);
  CacheArray<Node>::InitDistributedCacheMaster(
      nodes_channel_, config->nodes_block,
      size_t(nodes_mb_) * MEGABYTE, example_node,
      &nodes_);
  KdTreeHybridBuilder<Point, Node, Param>
      ::Build(module, *param, 0, config->n_points,
          &points_, &nodes_);
  fx_timer_stop(module, "tree");
}

#endif
