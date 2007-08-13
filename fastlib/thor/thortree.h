/**
 * @file thortree.h
 *
 * Support for trees in THOR.
 *
 * Mostly contains decomposition-related code.
 */

#ifndef SUPERPAR_KD_H
#define SUPERPAR_KD_H

#include "cachearray.h"
#include "thorstruct.h"

/**
 * A ThorTree is a distributed tree of any point and node type.
 *
 * A ThorTree encapsulates a distributed tree, by containing both of
 * its caches, its decomposition, and parameter object.
 */
template<typename TParam, typename TPoint, typename TNode>
class ThorTree {
  FORBID_COPY(ThorTree);

 public:
  typedef TParam Param;
  typedef TPoint Point;
  typedef TNode Node;
  typedef typename Node::Bound Bound;

 private:
  Param param_;
  ThorTreeDecomposition decomp_;
  DistributedCache *points_;
  DistributedCache *nodes_;

 public:
  ThorTree() {}
  ~ThorTree() {
    delete points_;
    delete nodes_;
  }

  void Init(const Param& param_in, const ThorTreeDecomposition &decomp_in,
      DistributedCache *points_in, DistributedCache *nodes_in) {
    param_.Copy(param_in);
    decomp_.Copy(decomp_in);
    points_ = points_in;
    nodes_ = nodes_in;
  }

  void set_param(const Param& param_in) {
    param_ = param_in;
  }
  void set_decomp(const ThorTreeDecomposition& decomp_in) {
    decomp_ = decomp_in;
  }
  void set_decomp(const ThorTreeDecomposition& decomp_in) {
    decomp_ = decomp_in;
  }

  /** Gets the parameter object characterizing this tree. */
  const Param& param() const { return param_; }
  /** Gets the parameter object characterizing this tree. */
  Param& param() { return param_; }
  /** Gets the decomposition dividing this tree. */
  const ThorTreeDecomposition& decomp() const { return decomp_; }
  /** Gets the array of points comporising this tree. */
  DistributedCache& points() { return *points_; }
  /** Gets the array of nodes dividing this tree. */
  DistributedCache& nodes() { return *nodes_; }

  /** Gets the root node. */
  const Node& root() const {
    // we cheat by using the tree decomposition as a cache of the node
    return decomp().root()->node();
  }
  /** Gets the number of points. */
  index_t n_points() const {
    return root_node().count();
  }

  /**
   * Updates each point and reaccumulates all bounds and statistics.
   */
  template<typename Result, typename Visitor>
  void Update(DistributedCache *results_cache, Visitor *visitor) {
    ThorUpdate<Param, Point, Node, Result, Visitor> updater;
    updater.Doit(rpc::rank(), &param_, decomp_,
        visitor, results_cache, &points_, &nodes_);
  }
};

//-------------------------------------------------------------------------
// IMPLEMENTATION
//-------------------------------------------------------------------------

//-- ThorTree

template<typename TParam, typename TPoint, typename TNode, typename Result>
void ThorKdTree<TParam, TPoint, TNode>::InitDistributedCacheMaster(
    int channel, const Result& default_result,
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

template<typename TParam, typename TPoint, typename TNode, typename Result>
void ThorKdTree<TParam, TPoint, TNode>::InitDistributedCacheWorker(
    int channel, size_t total_ram, DistributedCache *results) {
  DEBUG_ASSERT_MSG(rpc::rank() != 0, "Only workers call this");
  CacheArray<Result>::InitDistributedCacheWorker(channel, total_ram, results);
  results->StartSync();
  results->WaitSync();
}

template<typename TParam, typename TPoint, typename TNode, typename Result>
void ThorKdTree<TParam, TPoint, TNode>::InitDistributedCache(
    int channel, const Result& default_result,
    size_t total_ram, DistributedCache *results) {
  if (rpc::rank() == 0) {
    InitDistributedCacheMaster(channel, default_result, total_ram, results);
  } else {
    InitDistributedCacheWorker<Result>(channel, total_ram, results);
  }
}


#endif
