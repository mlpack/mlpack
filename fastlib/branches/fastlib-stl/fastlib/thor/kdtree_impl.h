/* Template implementations for kdtree.h. */

#include <armadillo>
#include "../base/arma_compat.h"

namespace thor {

/**
 * A generalized partition function for cached arrays.
 */
template<typename PartitionCondition, typename PointCache, typename Bound>
index_t Partition(
    PartitionCondition splitcond,
    index_t begin, index_t count,
    PointCache* points,
    Bound* left_bound, Bound* right_bound);

};

template<typename PartitionCondition, typename PointCache, typename Bound>
index_t thor::Partition(
    PartitionCondition splitcond,
    index_t begin, index_t count,
    PointCache* points,
    Bound* left_bound, Bound* right_bound) {
  static index_t processed_tot = 0;
  static index_t processed_cur = 0;

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
      arma::vec tmp;
      arma_compat::vectorToVec(left_v->vec(), tmp);
      if (!splitcond.is_left(left_v->vec())) {
        *right_bound |= tmp;
        break;
      }
      *left_bound |= tmp;
      left_i++;
      processed_cur++; //
    }

    for (;;) {
      if (unlikely(left_i > right_i)) return left_i;
      CacheRead<typename PointCache::Element> right_v(points, right_i);
      arma::vec tmp;
      arma_compat::vectorToVec(right_v->vec(), tmp);
      if (splitcond.is_left(right_v->vec())) {
        *left_bound |= tmp;
        break;
      }
      *right_bound |= tmp;
      right_i--;
      processed_cur++; //
    }

    points->Swap(left_i, right_i);

    DEBUG_ASSERT(left_i <= right_i);
    right_i--;
    processed_cur++; //

    //
    if (processed_cur > 10000000) {
      processed_tot += processed_cur;
      processed_cur = 0;
      NOTIFY("Partitioned %"LI"d points total...", processed_tot);
    }
    //
  }
}

template<typename TPoint, typename TNode, typename TParam>
void KdTreeHybridBuilder<TPoint, TNode, TParam>::Doit(
    struct datanode* module, const Param* param_in,
    index_t begin_index, index_t end_index,
    DistributedCache* points_inout, DistributedCache* nodes_create,
    TreeDecomposition* decomposition) {
  param_ = param_in;
  n_points_ = end_index - begin_index;

  points_.Init(points_inout, BlockDevice::M_MODIFY);
  nodes_.Init(nodes_create, BlockDevice::M_CREATE);

  index_t dimension;

  {
    CacheRead<Point> first_point(&points_, points_.begin_index());
    dimension = first_point->vec().length();
  }

  leaf_size_ = fx_param_int(module, "leaf_size", 32);
  block_size_ = points_.n_block_elems();
  if (leaf_size_ > block_size_) {
    NONFATAL("Decreasing leaf size from %d to %d due to block size!\n",
       int(leaf_size_), int(block_size_));
    leaf_size_ = block_size_;
  }
  chunk_size_ = fx_param_int(module, "chunk_size", leaf_size_ * 16);

  fx_timer_start(module, "tree_build");
  DecompNode* decomp_root;
  Bound bound(dimension);
  FindBoundingBox_(begin_index, end_index, &bound);
  Build_(begin_index, end_index, 0, rpc::n_peers(), bound, NULL, &decomp_root);
  decomposition->Init(decomp_root);
  fx_timer_stop(module, "tree_build");
}

template<typename TPoint, typename TNode, typename TParam>
void KdTreeHybridBuilder<TPoint, TNode, TParam>::FindBoundingBox_(
    index_t begin_index, index_t end_index, Bound* bound) {
  CacheReadIter<Point> point(&points_, begin_index);
  for (index_t i = end_index - begin_index; i--; point.Next()) {
    arma::vec tmp;
    arma_compat::vectorToVec(point->vec(), tmp);
    *bound |= tmp;
  }
}

template<typename TPoint, typename TNode, typename TParam>
index_t KdTreeHybridBuilder<TPoint, TNode, TParam>::Build_(
    index_t begin_col, index_t end_col,
    int begin_rank, int end_rank, const Bound& bound,
    Node* parent, DecompNode** decomp_pp) {
  index_t node_i = nodes_.AllocD(begin_rank, 1);
  Node* node = nodes_.StartWrite(node_i);
  DecompNode* left_decomp = NULL;
  DecompNode* right_decomp = NULL;

  node->set_range(begin_col, end_col - begin_col);
  node->bound().Reset();
  node->bound() |= bound;

  if (node->count() > leaf_size_) {
    index_t split_dim = BIG_BAD_NUMBER;
    double max_width = -1;

    // Short loop to find widest dimension
    for (index_t d = 0; d < node->bound().dim(); d++) {
      double w = node->bound()[d].width();

      if (unlikely(w > max_width)) {
        max_width = w;
        split_dim = d;
      }
    }

    DEBUG_ASSERT_MSG(max_width >= 0, "max_width = %f, dim = %"LI"d, n = %"LI"d",
        max_width, node->bound().dim(), node->count());

    // even if the max width is zero, we still* must* split it!
    Split_(node, begin_rank, end_rank, split_dim, parent,
        &left_decomp, &right_decomp);
  } else {
    node->set_leaf();
    // ensure leaves don't straddle block boundaries
    DEBUG_SAME_SIZE(node->begin() / points_.n_block_elems(),
        (node->end() - 1) / points_.n_block_elems());
    for (index_t i = node->begin(); i < node->end(); i++) {
      CacheRead<Point> point(&points_, i);
      node->stat().Accumulate(*param_, *point);
    }
  }

  if (parent != NULL) {
    // accumulate self to parent's statistics
    parent->stat().Accumulate(*param_,
        node->stat(), node->bound(), node->count());
  }

  node->stat().Postprocess(*param_, node->bound(), node->count());

  if (decomp_pp) {
    *decomp_pp = new DecompNode(
        typename TreeDecomposition::Info(begin_rank, end_rank),
        &nodes_, node_i, nodes_.end_index());
    DEBUG_ASSERT((left_decomp == NULL) == (right_decomp == NULL));
    if (left_decomp != NULL) {
      (*decomp_pp)->set_child(0, left_decomp);
      (*decomp_pp)->set_child(1, right_decomp);
    }
  }

  nodes_.StopWrite(node_i);

  return node_i;
}

template<typename TPoint, typename TNode, typename TParam>
void KdTreeHybridBuilder<TPoint, TNode, TParam>::Split_(
    Node* node, int begin_rank, int end_rank, int split_dim, Node *parent,
    DecompNode** left_decomp_pp, DecompNode** right_decomp_pp) {
  index_t split_col;
  index_t begin_col = node->begin();
  index_t end_col = node->end();
  int split_rank = (begin_rank + end_rank) / 2;
  double split_val;
  DRange current_range = node->bound()[split_dim];
  typename Node::Bound final_left_bound(node->bound().dim());
  typename Node::Bound final_right_bound(node->bound().dim());

  if ((node->begin() & points_.n_block_elems_mask()) == 0
      && (!parent || parent->begin() != node->begin())) {
    // We got one block of points!  Let's give away ownership.
    points_.cache()->GiveOwnership(
        points_.Blockid(node->begin()), begin_rank);
    // This is also a convenient time to display status.
    fl_print_progress("tree built",
        (node->begin() + block_size_) * 100 / n_points_);
  }

  if (1) {
    index_t goal_col;
    typename Node::Bound left_bound(node->bound().dim());
    typename Node::Bound right_bound(node->bound().dim());
    bool single_machine = (end_rank <= begin_rank + 1);

    if(single_machine) {
      // All points will go on the same machine, so do median split.
      goal_col = (begin_col + end_col) / 2;
    } else {
      // We're distributing these between machines.  Let's make sure
      // we give roughly even work to the machines.  What we do is
      // pretend the points are distributed as equally as possible, by
      // using the global number of machines and points, to avoid errors
      // introduced by doing this split computation recursively.
      goal_col = (uint64(split_rank) * 2 * n_points_ + rpc::n_peers())
          / rpc::n_peers() / 2;
    }

    if (node->count() > block_size_) {
      // Round the goal to the nearest block if larger than a block.
      goal_col = (goal_col + block_size_ / 2) / block_size_ * block_size_;
    }

    for (int iteration = 0;; iteration++) {
      // use linear interpolation to guess the value to split on.
      // this typically leads to convergence rather quickly.
      split_val = current_range.interpolate(
          (goal_col - begin_col) / double(end_col - begin_col));

      left_bound.Reset();
      right_bound.Reset();
      split_col = thor::Partition(
          HrectPartitionCondition(split_dim, split_val),
          begin_col, end_col - begin_col,
          &points_, &left_bound, &right_bound);

      if (node->count() < chunk_size_ && single_machine) {
        if (node->count() <= block_size_) {
          // Smaller than a block.
          // If it's not pathological, we'll finish off with a midpoint split.
          // Otherwise, we'll arbitrarily split up the block at the median.
          if (current_range.width() != 0) {
            goal_col = split_col;
          }
        } else if (iteration == 0) {
          // Larger than a block, so round to a block.
          DEBUG_ASSERT(end_col - begin_col == node->count());
          goal_col = (split_col + block_size_ / 2) / block_size_ * block_size_;
          if (goal_col <= begin_col) {
            goal_col += block_size_;
            DEBUG_ASSERT_MSG(goal_col < end_col - 1,
               "(%d %d) %d %d (%d %d)", node->begin(), begin_col, split_col, goal_col, node->end(), end_col);
          } else if (goal_col >= end_col - 1) {
            goal_col -= block_size_;
          }
        }
      }

      if (split_col == goal_col) {
        final_left_bound |= left_bound;
        final_right_bound |= right_bound;
        break;
      } else if (split_col < goal_col) {
        final_left_bound |= left_bound;
        current_range = right_bound[split_dim];
        if (current_range.width() == 0) {
          break; // identical elements
        }
        begin_col = split_col;
      } else if (split_col > goal_col) {
        final_right_bound |= right_bound;
        current_range = left_bound[split_dim];
        if (current_range.width() == 0) {
          break; // identical elements
        }
        end_col = split_col;
      }
    }
    if (split_col != goal_col) {
      // we got identical elements in that dimension, compute actual bound
      FindBoundingBox_(begin_col, goal_col, &final_left_bound);
      FindBoundingBox_(goal_col, end_col, &final_right_bound);
    }
    split_col = goal_col;
  }

  if (end_rank - begin_rank <= 1) {
    // I'm only one machine, don't need to expand children.
    left_decomp_pp = right_decomp_pp = NULL;
  }

  node->set_child(0, Build_(node->begin(), split_col,
      begin_rank, split_rank, final_left_bound, node, left_decomp_pp));
  node->set_child(1, Build_(split_col, node->end(),
      split_rank, end_rank, final_right_bound, node, right_decomp_pp));
}

template<typename Point, typename Node, typename Param>
void thor::CreateKdTreeMaster(const Param& param,
    int nodes_channel, int block_size_kb, double megs, datanode *module,
    index_t n_points,
    DistributedCache *points_cache, DistributedCache *nodes_cache,
    ThorTreeDecomposition<Node> *decomposition) {
  Node example_node;

  example_node.stat().Init(param);
  Point example_point;
  CacheArray<Point>::GetDefaultElement(points_cache, &example_point);
  example_node.bound().SetSize(example_point.vec().length());

  CacheArray<Node>::CreateCacheMaster(nodes_channel,
      CacheArray<Node>::ConvertBlockSize(example_node, block_size_kb),
      example_node, megs, nodes_cache);
  KdTreeHybridBuilder<Point, Node, Param> builder;
  builder.Doit(module, &param, 0, n_points, points_cache, nodes_cache,
      decomposition);
}

template<typename Point, typename Node, typename Param>
void thor::CreateKdTree(const Param& param,
    int nodes_channel, int extra_channel,
    datanode *module, index_t n_points,
    DistributedCache *points_cache,
    ThorTree<Param, Point, Node> *tree_out) {
  double megs = fx_param_double(module, "megs", 1000);
  DistributedCache *nodes_cache = new DistributedCache();
  Broadcaster<ThorTreeDecomposition<Node> > broadcaster;

  if (rpc::is_root()) {
    ThorTreeDecomposition<Node> decomposition;
    int block_size_kb = fx_param_int(module, "block_size_kb", 64);
    CreateKdTreeMaster<Point, Node>(param,
        nodes_channel, block_size_kb, megs, module, n_points,
        points_cache, nodes_cache, &decomposition);
    broadcaster.SetData(decomposition);
  } else {
    CacheArray<Node>::CreateCacheWorker(nodes_channel, megs, nodes_cache);
  }

  points_cache->Sync();
  nodes_cache->Sync();
  broadcaster.Doit(extra_channel); // broadcast the decomposition

  tree_out->Init(param, broadcaster.get(), points_cache, nodes_cache);
}
