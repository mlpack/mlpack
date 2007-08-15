/* Template implementations for thortree.h. */

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

template<typename TParam, typename TPoint, typename TNode>
template<typename Result, typename Visitor>
void ThorTree<TParam, TPoint, TNode>::Update(
    DistributedCache *results_cache, Visitor *visitor) {
  ThorUpdate<Param, Point, Node, Result, Visitor> updater;
  updater.Doit(rpc::rank(), &param_, decomp_,
      visitor, results_cache, points_, nodes_);
}

//-- ThorTreeDecomposition

template<class TNode>
void ThorTreeDecomposition<TNode>::FillLinearization_(DecompNode *node) {
  TreeGrain *grain;

  if (node->info().is_singleton() || !node->is_complete()) {
    grain = &grain_by_owner_[node->info().begin_rank];
    grain->node_index = node->index();
    grain->node_end_index = node->end_index();
    grain->point_begin_index = node->node().begin();
    grain->point_end_index = node->node().end();
  } else {
    for (int k = 0; k < Node::CARDINALITY; k++) {
      FillLinearization_(node->child(k));
    }
  }
}

//-- ThorUpdate

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
  node->bound().Reset();

  if (!node->is_leaf()) {
    for (int k = 0; k < Node::CARDINALITY; k++) {
      Recurse_(node->child(k), node);
    }
  } else {
    CacheWriteIter<Result> result(results_, node->begin());
    CacheWriteIter<Point> point(points_, node->begin());

    for (index_t i = node->begin(); i < node->end();
        i++, point.Next(), result.Next()) {
      visitor_->Update(i, point, result);
      node->stat().Accumulate(*param_, *point);
      node->bound() |= point->vec();
    }
  }

  if (likely(parent != NULL)) {
    parent->stat().Accumulate(
        *param_, node->stat(), node->bound(), node->count());
    parent->bound() |= node->bound();
    node->stat().Postprocess(*param_, node->bound(), node->count());
  }
}
