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

template<typename PartitionCondition, typename PointCache, typename Bound>
index_t Partition(
    PartitionCondition splitcond,
    index_t begin, index_t count,
    PointCache *points,
    Bound* left_bound, Bound* right_bound) {
  index_t left_i = begin;
  index_t right_i = begin + count - 1;

  /* At any point:
   *   every thing that strictly precedes left_i is correct
   *   every thing that strictly succeeds right_i is correct
   */
  for (;;) {
    for (;;) {
      if (unlikely(left_i > right_i)) return left_i;
      CacheRead<typename PointCache::Element> left_v(points, left_i);
      if (!splitcond.is_left(left_v->vec())) {
        *right_bound |= left_v->vec();
        break;
      }
      *left_bound |= left_v->vec();
      left_i++;
    }

    for (;;) {
      if (unlikely(left_i > right_i)) return left_i;
      CacheRead<typename PointCache::Element> right_v(points, right_i);
      if (splitcond.is_left(right_v->vec())) {
        *left_bound |= right_v->vec();
        break;
      }
      *right_bound |= right_v->vec();
      right_i--;
    }

    points->Swap(left_i, right_i);

    DEBUG_ASSERT(left_i <= right_i);
    right_i--;
  }

  abort();
}

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
  typedef ThorTreeDecomposition<Node> TreeDecomposition;
  typedef typename TreeDecomposition::DecompNode DecompNode;

 private:
  struct HrectPartitionCondition {
    int dimension;
    double value;

    HrectPartitionCondition(int dimension_in, double value_in)
      : dimension(dimension_in)
      , value(value_in) {}

    bool is_left(const Vector& vector) const {
      return vector.get(dimension) < value;
    }
  };

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
  void Doit(
      struct datanode *module,
      const Param* param_in_,
      index_t begin_index,
      index_t end_index,
      DistributedCache *points_inout,
      DistributedCache *nodes_create,
      TreeDecomposition *decomposition) {
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
    chunk_size_ = points_.n_block_elems();

    fx_timer_start(module, "tree_build");
    Build_(decomposition);
    fx_timer_stop(module, "tree_build");
  }

  void FindBoundingBox_(index_t begin, index_t count, Bound *bound);
  DecompNode *Build_(index_t node_i, int begin_rank, int end_rank,
      const Bound& bound, Node *parent);
  void Build_(TreeDecomposition *decomposition);
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
typename KdTreeHybridBuilder<TPoint, TNode, TParam>::DecompNode *
KdTreeHybridBuilder<TPoint, TNode, TParam>::Build_(
    index_t node_i, int begin_rank, int end_rank, const Bound& bound,
    Node *parent) {
  // TODO: Thorlit this into smaller functions!  This keeps on growing
  // and growing... Proposal: (1) a non-leaf subfunction, (2) a function
  // responsible for the quantile-find
  Node *node = nodes_.StartWrite(node_i);
  bool leaf = true;
  DecompNode *left_decomp = NULL;
  DecompNode *right_decomp = NULL;
  bool single_machine = (end_rank <= begin_rank + 1);

  node->bound().Reset();
  node->bound() |= bound;

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
      ThorRange current_range = node->bound().get(split_dim);

      if (index_t(node->count()) == index_t(points_.n_block_elems())) {
        // We got one block of points!  Let's give away ownership.
        // This is also a convenient time to display status.
        // TODO: There's a slight bug -- the last block might not be a
        // "full" block and its ownership will be given to the first
        // machine even though this is incorrect.
        percent_indicator("tree built", node->end(), n_points_);
        points_.cache()->GiveOwnership(
            points_.Blockid(node->begin()),
            begin_rank);
      }

      if (node->count() <= chunk_size_) {
        // perform a midpoint split
        split_val = current_range.mid();
        split_col = Partition(
            HrectPartitionCondition(split_dim, split_val),
            begin_col, end_col - begin_col,
            &points_, &final_left_bound, &final_right_bound);
      } else {
        index_t goal_col;
        typename Node::Bound left_bound;
        typename Node::Bound right_bound;
        left_bound.Init(dim_);
        right_bound.Init(dim_);

        if (single_machine) {
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
          split_col = Partition(
              HrectPartitionCondition(split_dim, split_val),
              begin_col, end_col - begin_col,
              &points_, &left_bound, &right_bound);

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
      left->set_range(node->begin(), split_col - node->begin());
      left_decomp = Build_(left_i, begin_rank, split_rank,
          final_left_bound, node);
      nodes_.StopWrite(left_i);

      index_t right_i = nodes_.AllocD(split_rank);
      Node *right = nodes_.StartWrite(right_i);
      right->set_range(split_col, node->end() - split_col);
      right_decomp = Build_(right_i, split_rank,
          single_machine ? split_rank : end_rank, final_right_bound, node);
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

  if (parent) {
    parent->stat().Accumulate(*param_,
        node->stat(), node->bound(), node->count());
  }
  node->stat().Postprocess(*param_, node->bound(), node->count());

  // now store tree decomposition
  DecompNode *decomp = NULL;

  if (unlikely(end_rank > begin_rank)) {
    decomp = new DecompNode(
        typename TreeDecomposition::Info(begin_rank, end_rank),
        &nodes_, node_i, nodes_.end_index());
    DEBUG_ASSERT((left_decomp == NULL) == (right_decomp == NULL));
    if (left_decomp != NULL) {
      decomp->set_child(0, left_decomp);
      decomp->set_child(1, right_decomp);
    }
    decomp->info().begin_rank = begin_rank;
    decomp->info().end_rank = end_rank;
  } else {
    DEBUG_ASSERT(left_decomp == NULL);
    DEBUG_ASSERT(right_decomp == NULL);
  }

  nodes_.StopWrite(node_i);

  return decomp;
}

template<typename TPoint, typename TNode, typename TParam>
void KdTreeHybridBuilder<TPoint, TNode, TParam>::Build_(
    TreeDecomposition *decomposition) {
  int begin_rank = 0;
  int end_rank = rpc::n_peers();
  index_t node_i = nodes_.AllocD(begin_rank);
  Node *node = nodes_.StartWrite(node_i);

  DEBUG_SAME_INT(node_i, 0);

  node->set_range(begin_index_, end_index_);

  Bound bound;
  bound.Init(dim_);
  FindBoundingBox_(node->begin(), node->end(), &bound);

  DecompNode *decomp_root = Build_(node_i, begin_rank, end_rank,
      bound, NULL);
  if (decomposition) {
    decomposition->Init(decomp_root);
  } else {
    delete decomp_root;
  }

  nodes_.StopWrite(node_i);
}

//--------------------------------------------------------------------------

/**
 * A parallelizable tree-updater.
 *
 * What this is useful for is applying QResults to update query points.
 * Also, the visitor itself gets to accumulate information over the
 * tree itself for its own purposes.
 */
template<class TParam, class TPoint, class TNode, class TResult, class TVisitor>
class ThorUpdate {
 public:
  typedef TParam Param;
  typedef TPoint Point;
  typedef TNode Node;
  typedef TResult Result;
  typedef TVisitor Visitor;
  typedef ThorTreeDecomposition<Node> TreeDecomposition;
  typedef typename TreeDecomposition::DecompNode DecompNode;

 private:
  CacheArray<Point> *points_;
  CacheArray<Node> *nodes_;
  CacheArray<Result> *results_;
  Visitor *visitor_;
  const Param *param_;

 public:
  /**
   * Perform the tree update for just one machine.
   *
   * Note that this will *not* perform a reduction.
   *
   * See class-level comments.
   */
  void Doit(int my_rank, const Param *param, const TreeDecomposition& decomp,
      Visitor *visitor, DistributedCache *results_cache,
      DistributedCache *points_cache, DistributedCache *nodes_cache);

 private:
  void Assemble_(const DecompNode *decomp, Node *parent);
  void Recurse_(index_t node_index, Node *parent);
};

template<class TParam, class TPoint, class TNode, class TResult, class TVisitor>
void ThorUpdate<TParam, TPoint, TNode, TResult, TVisitor>::Doit(
    int my_rank, const Param *param, const TreeDecomposition& decomp,
    Visitor *visitor, DistributedCache *results_cache,
    DistributedCache *points_cache, DistributedCache *nodes_cache) {
  bool is_main_machine = (my_rank == decomp.root()->info().begin_rank);

  param_ = param;
  visitor_ = visitor;

  // Find my machine by searching the tree
  TreeGrain my_grain = decomp.grain_by_owner(my_rank);

  if (my_grain.is_valid()) {
    // I get a work item (this is always the case unlses for some reason
    // the tree is incredibly small)
    results_ = new CacheArray<Result>();
    results_->Init(results_cache, BlockDevice::M_MODIFY,
        my_grain.point_begin_index, my_grain.point_end_index);
    points_ = new CacheArray<Point>();
    points_->Init(points_cache, BlockDevice::M_MODIFY,
        my_grain.point_begin_index, my_grain.point_end_index);
    nodes_ = new CacheArray<Node>();
    nodes_->Init(nodes_cache, BlockDevice::M_MODIFY,
        my_grain.node_index, my_grain.node_end_index);
    Recurse_(my_grain.node_index, NULL);
    delete results_;
    delete nodes_;
    delete points_;
  }

  nodes_ = NULL;
  points_ = NULL;
  results_ = NULL;

  nodes_cache->StartSync();
  points_cache->StartSync();
  nodes_cache->WaitSync();

  if (is_main_machine) {
    // I'm the master machine!  I update the top part of the tree! WOOT!
    nodes_ = new CacheArray<Node>();
    // NOTE: M_APPEND actually means that writes to blocks are exclusive
    nodes_->Init(nodes_cache, BlockDevice::M_APPEND);
    Assemble_(decomp.root(), NULL);
    delete nodes_;
  }

  nodes_cache->StartSync();
  nodes_cache->WaitSync();
  points_cache->WaitSync();
}

template<class TParam, class TPoint, class TNode, class TResult, class TVisitor>
void ThorUpdate<TParam, TPoint, TNode, TResult, TVisitor>::Assemble_(
    const DecompNode *decomp, Node *parent) {
  // We're at a leaf in the decomposition tree.  Just update our parent.
  CacheWrite<Node> node(nodes_, decomp->index());
  DEBUG_ASSERT(decomp->index() >= 0);

  if (decomp->is_complete() && !decomp->info().is_singleton()) {
    // The node has children in the decomposition tree, so they haven't been
    // updated.
    node->stat().Reset(*param_);

    for (int k = 0; k < Node::CARDINALITY; k++) {
      Assemble_(decomp->child(k), node);
    }
  }

  if (likely(parent != NULL)) {
    parent->stat().Accumulate(*param_,
        node->stat(), node->bound(), node->count());
  }

  node->stat().Postprocess(*param_, node->bound(), node->count());
}

template<class TParam, class TPoint, class TNode, class TResult, class TVisitor>
void ThorUpdate<TParam, TPoint, TNode, TResult, TVisitor>::Recurse_(
    index_t node_index, Node *parent) {
  CacheWrite<Node> node(nodes_, node_index);

  node->stat().Reset(*param_);

  if (!node->is_leaf()) {
    for (int k = 0; k < Node::CARDINALITY; k++) {
      Recurse_(node->child(k), node);
    }
  } else {
    CacheWriteIter<Result> result(results_, node->begin());
    CacheWriteIter<Point> point(points_, node->begin());

    for (index_t i = 0; i < node->count(); i++, point.Next(), result.Next()) {
      visitor_->Update(point, result);
      node->stat().Accumulate(*param_, *point);
    }
  }

  if (likely(parent != NULL)) {
    parent->stat().Accumulate(
        *param_, node->stat(), node->bound(), node->count());
    node->stat().Postprocess(*param_, node->bound(), node->count());
  }
}

//--------------------------------------------------------------------------

/**
 * A distributed kd-tree (works non-distributed too).
 */
template<typename TParam, typename TPoint, typename TNode>
class ThorKdTree {
 public:
  typedef TParam Param;
  typedef TPoint Point;
  typedef TNode Node;
  typedef ThorTreeDecomposition<Node> TreeDecomposition;
  typedef typename ThorTreeDecomposition<Node>::DecompNode DecompNode;

 private:
  struct Config {
    TreeDecomposition decomp;
    Param *param;
    int nodes_block;
    int points_block;
    int n_points;
    int dim;

    OT_DEF(Config) {
      OT_MY_OBJECT(decomp);
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
  Broadcaster<Config> config_broadcaster_;
  Config *config_;

 public:
  ThorKdTree() {}
  ~ThorKdTree() {}

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

  /**
   * Performs a distributed tree-update.
   *
   * Note that this does NOT broadcast or reduce the visitor -- you must
   * take care of it if you need to store data in the visitor!
   */
  template<typename Result, typename Visitor>
  void Update(
      DistributedCache *results_cache, Visitor *visitor) {
    ThorUpdate<Param, Point, Node, Result, Visitor> updater;
    updater.Doit(rpc::rank(), &param(), config_->decomp,
        visitor, results_cache, &points_, &nodes_);
  }

  /**
   * Creates a cache that's suitable for storing results or anything else.
   *
   * This array will be decomposed in the exact same way the points
   * array is decomposed (same block size and everything).
   */
  template<typename Result>
  void InitDistributedCacheMaster(int channel, const Result& default_result,
      size_t total_ram, DistributedCache *results) {
    DEBUG_ASSERT_MSG(rpc::rank() == 0, "Only master calls this");
    index_t block_size = config_->points_block;
    CacheArray<Result>::InitDistributedCacheMaster(channel,
        block_size, default_result, total_ram, results);

    for (int i = 0; i < rpc::n_peers(); i++) {
      const TreeGrain *grain = &config_->decomp.grain_by_owner(i);
      if (grain->is_valid()) {
        BlockDevice::blockid_t begin_block =
            (grain->point_begin_index + block_size - 1) / block_size;
        BlockDevice::blockid_t end_block =
            (grain->point_end_index + block_size - 1) / block_size;
        DEBUG_ASSERT(results->n_blocks() == begin_block);
        results->AllocBlocks(end_block - begin_block, i);
      }
    }

    results->StartSync();
    results->WaitSync();
  }

  /**
   * The worker version of create-a-new-distributed-cache.
   */
  template<typename Result>
  void InitDistributedCacheWorker(int channel, size_t total_ram,
      DistributedCache *results) {
    DEBUG_ASSERT_MSG(rpc::rank() != 0, "Only workers call this");
    CacheArray<Result>::InitDistributedCacheWorker(channel, total_ram, results);
    results->StartSync();
    results->WaitSync();
  }

  DistributedCache& points() { return points_; }
  DistributedCache& nodes() { return nodes_; }

  /**
   * Gets the known parameter object.
   *
   * This object is not updated except when the tree is created.
   * If you need to update this, then use a broadcaster to relay the
   * new param object between machines, and use set_param!
   */
  Param& param() const {
    return *config_->param;
  }

  /**
   * Sets the parameter objct.
   */
  void set_param(Param &new_param) {
    delete *config_->param;
    config_->param = new Param(new_param);
  }

  index_t nodes_block() const { return config_->nodes_block; }
  index_t points_block() const { return config_->points_block; }
  index_t n_points() const { return config_->n_points; }

  ThorTreeDecomposition<Node>& decomposition() const {
    return config_->decomp;
  }

 private:
  void MasterLoadData_(
      Config *config, Param *param, int tag, datanode *module);
  void MasterBuildTree_(
      Config *config, Param *param, int tag, datanode *module);
};

template<typename TParam, typename TPoint, typename TNode>
void ThorKdTree<TParam, TPoint, TNode>::Init(Param **parampp, int param_tag,
    int base_channel, datanode *module) {
  points_module_ = fx_submodule(module, "points", "points");
  nodes_module_ = fx_submodule(module, "nodes", "nodes");
  points_channel_ = base_channel + 0;
  nodes_channel_ = base_channel + 1;
  points_mb_ = fx_param_int(points_module_, "mb", 2000);
  nodes_mb_ = fx_param_int(nodes_module_, "mb", 500);
  if (rpc::is_root()) {
    Config config;
    MasterLoadData_(&config, *parampp, param_tag, module);
    MasterBuildTree_(&config, *parampp, param_tag, module);
    config.param = *parampp;
    config_broadcaster_.SetData(config);
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
  config_broadcaster_.Doit(base_channel + 2);
  config_ = &config_broadcaster_.get();
  delete *parampp;
  *parampp = new Param(*config_->param);
}

template<typename TParam, typename TPoint, typename TNode>
void ThorKdTree<TParam, TPoint, TNode>::MasterLoadData_(
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
      points_channel_, config->points_block, default_point,
      size_t(points_mb_) * MEGABYTE,
      &points_);
  CacheArray<Point> points_array;
  points_array.Init(&points_, BlockDevice::M_CREATE);
  index_t i = 0;

  for (;;) {
    i = points_array.AllocD(rpc::rank(), 1);
    CacheWrite<Point> point(&points_array, i);
    bool is_done;
    success_t rv = schema.ReadPoint(&reader, point->vec().ptr(), &is_done);
    param->SetPointExtras(tag, i, point);
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
void ThorKdTree<TParam, TPoint, TNode>::MasterBuildTree_(
    Config *config, Param *param, int tag, datanode *module) {
  config->nodes_block = fx_param_int(nodes_module_, "block", 256);

  fprintf(stderr, "master: Building tree\n");
  fx_timer_start(module, "tree");
  Node example_node;
  example_node.Init(config->dim, *param);
  CacheArray<Node>::InitDistributedCacheMaster(
      nodes_channel_, config->nodes_block, example_node,
      size_t(nodes_mb_) * MEGABYTE,
      &nodes_);
  KdTreeHybridBuilder<Point, Node, Param> builder;
  builder.Doit(module, param, 0, config->n_points,
          &points_, &nodes_, &config->decomp);
  fx_timer_stop(module, "tree");
}

#endif
