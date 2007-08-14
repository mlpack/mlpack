/**
 * @file thortree.h
 *
 * Support for trees in THOR.
 *
 * Mostly contains decomposition-related code.
 */

#ifndef SUPERPAR_KD_H
#define SUPERPAR_KD_H

#include "thortree_algs.h"

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
  ThorTreeDecomposition<Node> decomp_;
  DistributedCache *points_;
  DistributedCache *nodes_;

 public:
  ThorTree() {}
  ~ThorTree() {
    delete points_;
    delete nodes_;
  }

  void Init(const Param& param_in,
      const ThorTreeDecomposition<Node> &decomp_in,
      DistributedCache *points_in, DistributedCache *nodes_in) {
    param_.Copy(param_in);
    decomp_.Copy(decomp_in);
    points_ = points_in;
    nodes_ = nodes_in;
  }

  void set_param(const Param& param_in) {
    param_ = param_in;
  }
  void set_decomp(const ThorTreeDecomposition<Node>& decomp_in) {
    decomp_ = decomp_in;
  }

  /** Gets the parameter object characterizing this tree. */
  const Param& param() const { return param_; }
  /** Gets the parameter object characterizing this tree. */
  Param& param() { return param_; }
  /** Gets the decomposition dividing this tree. */
  const ThorTreeDecomposition<Node>& decomp() const { return decomp_; }
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
    return root().count();
  }

  /**
   * Updates each point and reaccumulates all bounds and statistics.
   */
  template<typename Result, typename Visitor>
  void Update(DistributedCache *results_cache, Visitor *visitor) {
    ThorUpdate<Param, Point, Node, Result, Visitor> updater;
    updater.Doit(rpc::rank(), &param_, decomp_,
        visitor, results_cache, points_, nodes_);
  }

  /**
   * Creates a new cache that has one element per point in the original
   * cache, distributed among the machines in the same way the points
   * are.
   *
   * This automatically calls the master or worker version of this depending
   * on rank.
   */
  template<typename Result>
  void CreateResultCache(int channel, const Result& default_result,
      double megs, DistributedCache *results);
  /**
   * Same as CreateResultCache, but only the master calls this.
   *
   * This actually does the initialization.
   */
  template<typename Result>
  void CreateResultCacheMaster(int channel, const Result& default_result,
      double megs, DistributedCache *results);
  /**
   * Same as CreateResultCache, but only the workers calls this.
   *
   * This just waits for the master to sync up.
   */
  template<typename Result>
  void CreateResultCacheWorker(int channel,
      double megs, DistributedCache *results);
};

//-------------------------------------------------------------------------
// IMPLEMENTATION
//-------------------------------------------------------------------------

//-- ThorTree

template<typename TParam, typename TPoint, typename TNode>
template<typename Result>
void ThorTree<TParam, TPoint, TNode>::CreateResultCacheMaster(
    int channel, const Result& default_result,
    double megs, DistributedCache *results) {
  DEBUG_ASSERT_MSG(rpc::rank() == 0, "Only master calls this");
  index_t block_size = CacheArray<Point>::GetNumBlockElements(points_);
  CacheArray<Result>::CreateCacheMaster(channel,
      block_size, default_result, megs, results);

  for (int i = 0; i < rpc::n_peers(); i++) {
    const TreeGrain *grain = &decomp_.grain_by_owner(i);
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

template<typename TParam, typename TPoint, typename TNode>
template<typename Result>
void ThorTree<TParam, TPoint, TNode>::CreateResultCacheWorker(
    int channel, double megs, DistributedCache *results) {
  DEBUG_ASSERT_MSG(rpc::rank() != 0, "Only workers call this");
  CacheArray<Result>::CreateCacheWorker(channel, megs, results);
  results->StartSync();
  results->WaitSync();
}

template<typename TParam, typename TPoint, typename TNode>
template<typename Result>
void ThorTree<TParam, TPoint, TNode>::CreateResultCache(
    int channel, const Result& default_result,
    double megs, DistributedCache *results) {
  if (rpc::rank() == 0) {
    CreateResultCacheMaster(channel, default_result, megs, results);
  } else {
    CreateResultCacheWorker<Result>(channel, megs, results);
  }
}

#endif
