/**
 * @file thortree.h
 *
 * Support for trees in THOR.
 *
 * Mostly contains decomposition-related code.
 */

#ifndef THOR_THORTREE_H
#define THOR_THORTREE_H

#include "thor_struct.h"

/**
 * A single component of the tree, viewed as a work item.
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

  template<typename TSkeletonNode>
  void Init(const TSkeletonNode& node) {
    node_index = node.index();
    node_end_index = node.end_index();
    point_begin_index = node.node().begin();
    point_end_index = node.node().end();
  }

  void InitInvalid() {
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
      return math::RoundInt(double(end_rank - begin_rank)
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
    DEBUG_ASSERT(root_->info().end_rank == rpc::n_peers());
    DEBUG_ASSERT(root_in != NULL);
    grain_by_owner_.Init(rpc::n_peers());
    for (int i = 0; i < grain_by_owner_.size(); i++) {
      grain_by_owner_[i].InitInvalid();
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
 * A ThorTree is a distributed tree of any point and node type.
 *
 * A ThorTree encapsulates a distributed tree, by containing both of
 * its caches, its decomposition, and parameter object.
 */
template<typename TParam, typename TPoint, typename TNode>
class ThorTree {
  FORBID_ACCIDENTAL_COPIES(ThorTree);

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
    param_.InitCopy(param_in);
    decomp_.InitCopy(decomp_in);
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
  void Update(DistributedCache *results_cache, Visitor *visitor);

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
template<class TParam, class TPoint, class TNode,
         class TResult, class TVisitor>
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

#include "thortree_impl.h"

#endif
