/**
 * @param thortree_algs.h
 *
 * Tree algorithms related to THOR trees.
 *
 * These primarily have to do with decomposition.
 */

#ifndef THORTREE_ALGS_H
#define THORTREE_ALGS_H

/**
 * A single work item.
 */
struct TreeGrain {
  /** The root node index, also the first node in the contiguous sequence. */
  index_t node_index;
  /** One past the last node in the contiguous node sequence. */
  index_t node_end_index;
  /** The first point. */
  index_t point_begin_index;
  /** One past the last point. */
  index_t point_end_index;

  void InitBlank() {
    node_index = -1;
    node_end_index = -1;
    point_begin_index = -1;
    point_end_index = -1;
  }

  bool is_valid() const {
    return node_index >= 0;
  }

  index_t n_points() const {
    return point_end_index - point_begin_index;
  }
  index_t n_nodes() const {
    return node_end_index - node_index;
  }

  OT_DEF_BASIC(TreeGrain) {
    OT_MY_OBJECT(node_index);
    OT_MY_OBJECT(node_end_index);
    OT_MY_OBJECT(point_begin_index);
    OT_MY_OBJECT(point_end_index);
  }
};

/**
 * A distributed decomposition of an ThorTree.
 *
 * Currently only handles trivial node-to-machine decompositions.
 */
template<typename TNode>
class ThorTreeDecomposition {
 public:
  typedef TNode Node;
  struct Info {
    int begin_rank;
    int end_rank;

    Info(int begin_rank_in, int end_rank_in)
        : begin_rank(begin_rank_in), end_rank(end_rank_in) {}

    bool is_singleton() const {
      return end_rank - begin_rank <= 1;
    }

    bool contains(int rank) const {
      return rank >= begin_rank && rank < end_rank;
    }

    int Interpolate(index_t numerator, index_t denominator) {
      return math::IntRound(double(end_rank - begin_rank)
          * numerator / denominator) + begin_rank;
    }

    void Fix() {
      if (end_rank <= begin_rank) {
        end_rank = begin_rank + 1;
      }
    }

    OT_DEF_BASIC(Info) {
      OT_MY_OBJECT(begin_rank);
      OT_MY_OBJECT(end_rank);
    }
  };

  typedef ThorSkeletonNode<Node, Info> DecompNode;

 private:
  /**
   * The tree decomposition.
   */
  DecompNode *root_;
  ArrayList<TreeGrain> grain_by_owner_;

  OT_DEF(ThorTreeDecomposition) {
    OT_PTR(root_);
    OT_MY_OBJECT(grain_by_owner_);
  }

 public:
  /**
   * Initializes given the root of a decomposition tree.
   */
  void Init(DecompNode *root_in) {
    root_ = root_in;
    DEBUG_ASSERT(root_->info().begin_rank == 0);
    int n = root_->info().end_rank;
    grain_by_owner_.Init(n);
    for (int i = 0; i < n; i++) {
      grain_by_owner_[i].InitBlank();
    }
    FillLinearization_(root_);
  }

  /** Returns the root of the decomposition tree. */
  DecompNode *root() const {
    return root_;
  }

  /**
   * Gets the portion of the tree that is assigned to the specified machine.
   */
  const TreeGrain& grain_by_owner(int rank) const {
    return grain_by_owner_[rank];
  }

 private:
  void FillLinearization_(DecompNode *node);
};

/**
 * A parallelizable tree-updater.
 *
 * You use this to read from a separate array of results and update each
 * query point, simultaneously updating all the statistics in the tree.
 *
 * This is only parallel so that each machine only works on its local data,
 * avoiding communication.  Since updating is not CPU intensive this is
 * in most cases an I/O bound process, so it is only single-threaded.
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
   * Note that this will *not* perform any reductions on the visitor, each
   * machine has its separate visitor for its own part of the tree.
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

//--------------------------------------------------------------------------
// IMPLEMENTATION
//--------------------------------------------------------------------------

//-- ThorTreeDecomposition

template<class TNode> {
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

    for (index_t i = 0; i < node->count(); i++, point.Next(), result.Next()) {
      visitor_->Update(point, result);
      node->stat().Accumulate(*param_, *point);
      node->bound() |= *point;
    }
  }

  if (likely(parent != NULL)) {
    parent->stat().Accumulate(
        *param_, node->stat(), node->bound(), node->count());
    parent->bound() |= node->bound();
    node->stat().Postprocess(*param_, node->bound(), node->count());
  }
}

#endif
