/**
 * @file gnp.h
 *
 * Framework for defining and executing generalized n-body problems, in
 * serial or parallel.
 */

#ifndef SUPERPAR_GNP_H
#define SUPERPAR_GNP_H



/*

*/


// BIG BIG TODO:
//  - Storage solution for GNP's
//  - Per reference node WEIGHT is needed for matrix vector multiplication
//    - RData type
//    - problem: summary statistics require RData
//    - QData too?
//  - Would a dual-tree algorithm want to know count information?
//  - Figure out how to turn things completely off
//  - Can mass results be used for pruning?

template<
  typename TAlgorithm,
  typename TPoint, typename TBound,
  typename TRInfo, typename TRStat,
  typename TQInfo, typename TQStat, typename QResultStat, typename TQDelta,
  typename TGlobalStat
>
class GnpDualTree {
 public:
  /** The dual-tree parts of the algorithm. */
  typedef TAlgorithm Algorithm;

  /** The type of point considered. */
  typedef TPoint Point;
  /** The bounding type used for pruning and tree construction. */
  typedef TBound Bound;

  /** Extra data associated with each reference point. */
  typedef TRInfo RInfo;
  /** Stat that is pre-computed for each reference node. */
  typedef TRStat RStat;

  /** Extra data associated with each query point. */
  typedef TQInfo QInfo;
  /** Stat that is pre-computed for each query node. */
  typedef TQStat QStat;

  /** A desired result for each query point. */
  typedef TQResult QResult;
  /** Summaries over results. */
  typedef TQResultStat QResultStat;
  /** Stat updated at an upper level and needs to be distributed to all
   * leaves eventually. */
  typedef TQDelta QDelta;

  /**
   * Stat computed for the entire computation.
   */
  typedef TGlobalStat GlobalStat;
};

template<
  class TGNP, //< A query-reference generalized N-body problem
  class TArray, //< The array type to use for the implementation
>
class GnpDualTreeRunner {
 public:
  /** The array type used to store data. */
  typedef TArray Array;
  /** The generalized N-body problem to solve. */
  typedef TGNP GNP;

#error "todo: typedefs"

  /** Tree nodes. */
  typedef SpNode< Bound > TreeNode;

 private:
  Algorithm algorithm_;

  Array<TreeNode> r_tree_a_;
  Array<Point> r_point_a_;
  Array<RStat> r_stat_a_;
  //Array<RData> r_data_a_;

  Array<TreeNode> q_tree_a_;
  Array<Point> q_point_a_;
  Array<QStat> q_stat_a_;
  Array<QResultStat> q_result_stat_a_;
  Array<Delta> q_delta_a_;

  Array<QResult> q_result_a_;

  GlobalStat global_stat_a_;

 public:
  void Init(datanode *module) {
    algorithm_.Init(fx_submodule(module, "algorithm", "algorithm"),
        &global_stat_a_);
    #error need to create kd trees
  }
  void Compute();

 private:
  void BaseCase_(
      const TreeNode *r_node,
      const RStat& r_stat,
      const TreeNode *q_node,
      const QStat& q_stat,
      QResultStat *q_result_stat) {
  void DualTree_(index_t q_node_i, index_t r_node_i);

  void DistributeDeltas_();

  void RecursivelyDistributeDeltas_(const TreeNode *q_node,
      Delta *q_delta);
};


template<class TGNP, class TArray>
void GnpDualTreeRunner<TGNP, TArray>::DistributeDeltas_() {
  if (HAVE_MASS_RESULT) {
    const TreeNode *q_root_node = q_tree_a_.StartRead(
        TreeNode::ROOT_INDEX);
    Delta *q_root_delta = q_delta_a_.StartWrite(
        TreeNode::ROOT_INDEX);

    RecursivelyDistributeDeltas_(q_root_node, q_root_delta);
    
    q_delta_a_.StopWrite(q_root_delta, TreeNode::ROOT_INDEX);
    q_tree_a_.StopRead(q_root_node, TreeNode::ROOT_INDEX);
  }
}

template<class TGNP, class TArray>
void GnpDualTreeRunner<TGNP, TArray>::RecursivelyDistributeDeltas_(
    const TreeNode *q_node, Delta *q_delta) {
  q_delta->StartDistribute();

  if (q_node->is_leaf()) {
    for (index_t i = 0; i < q_node->cardinality(); i++) {
      index_t q_child_i = 0;
      const TreeNode *q_child_node = q_tree_a_.StartRead(q_child_i);
      Delta *q_child_delta = q_delta_a_.StartWrite(q_child_i);

      q_delta->DistributeTo(q_child_delta);
      RecursivelyDistributeDeltas_(q_child_node, q_child_delta);

      q_tree_a_.StopRead(q_child_node, q_child_i);
      q_delta_a_.StopWrite(q_child_delta, q_child_i);
    }
  } else {
    for (index_t i = q_node->begin(); i < q_node->end(); i++) {
      QResult *q_result = q_result_a_.StartWrite(q_point_i);

      q_delta->DistributeTo(q_result);

      q_result_a_.StopWrite(q_result, q_point_i);
    }
  }

  q_delta->StopDistribute();
}

template<class TGNP, class TArray>
void GnpDualTreeRunner<TGNP, TArray>::BaseCase_(
    const TreeNode *r_node,
    const RStat& r_stat,
    const TreeNode *q_node,
    const QStat& q_stat,
    QResultStat *q_result_stat) {
  index_t q_begin = q_node->begin();
  index_t r_begin = r_node->begin();
  index_t q_point_i = q_node->count() + q_begin;
  index_t r_end = r_node->count() + r_begin;

  q_result_stat->StartUpdate();

  // TODO: Refresh

  do {
    index_t r_point_i = r_count + r_begin;

    q_point_i--;

    const Point *q_point = q_point_a_.StartRead(q_point_i);
    QResult *q_result = q_result_a_.StartWrite(q_point_i);
    // Copy the result to allow it to be register-allocated
    QResult q_result_cached(q_result);

    bool did_a_prune = algorithm_.TryPrune(
      r_node->bound(), r_stat,
      q_node->bound(), q_stat, q_result_stat, q_result,
      &g_stat);

    do {
      r_point_i--;
      // INNERMOST LOOP - PERFORMANCE CRUNCH HERE!
      // We might want fast sub-arrays of some sort
      const Point *r_point = r_point_a_.StartRead(r_point_i);
      q_result_cached.Update(*q_point, *r_point, r_point_i);
      r_point_a_.StopWrite(r_point, r_point_i);
    } while (likely(r_point_i > q_begin));

    *q_result = q_result_cached;
    q_result_a_.StopWrite(q_result, q_point_i);
    q_point_a_.StopRead(q_point, q_point_i);

    q_result_stat->Update(q_result_cached);
  } while (likely(q_point_i > q_begin));

  q_result_stat->StopUpdate();
}

template<class TGNP, class TArray>
void GnpDualTreeRunner<TGNP, TArray>::DualTree_(
    index_t q_node_i, index_t r_node_i) {
  // Read all data
  const TreeNode *r_node = r_tree_a_.StartRead(r_node_i);
  const RStat *r_stat = r_stat_a_.StartRead(r_node_i);

  const TreeNode *q_node = q_tree_a_.StartRead(q_node_i);
  const QStat *q_stat = q_stat_a_.StartRead(q_node_i);
  QResultStat *q_result_stat = q_result_stat_a_.StartWrite(q_node_i);
  Delta *q_delta = q_delta_a_.StartWrite(q_node_i);

  bool did_a_prune = algorithm_.TryPrune(
      r_node->bound(), *r_stat,
      q_node->bound(), *q_stat, q_result_stat, q_delta,
      &global_stat_);

  if (!did_a_prune) {
    if (q_node->is_leaf() && r_node->is_leaf()) {
      BaseCase(r_node, r_stat, q_node, q_stat, q_result_stat);
    } else if ((q_node->count() >= r_node->count() && !q_node->is_leaf())
        || (r_node->is_leaf())) {
      q_result_stat->StartUpdate();
      for (index_t i = 0; i < q_node->cardinality(); i++) {
        index_t q_child_i = q_node->child(i);
        GnpQueryReferenceRunner::DualTree(q_child_i, r_node_i);
        const QResultStat *q_child_result_stat = q_result_stat_a_.StartRead(q_child_i);
        // TODO: Refresh
        q_result_stat->Update(q_child_result_stat);
        q_result_stat_a_.StopRead(q_child_result_stat, q_child_i);
      }
      q_result_stat->StopUpdate();
    } else {
      DEBUG_ASSERT(node.cardinality() == 2);
      double priority[2];
      index_t r_child_order[2];

      for (index_t i = 0; i < node.cardinality(); i++) {
        index_t r_child_i = r_node->child(i);
        const TreeNode *r_child_node = r_tree_a_.StartRead(r_child_i);
        const RStat *r_child_stat = r_stat_a_.StartRead(r_child_i);

        r_child_order[i] = r_child_i;

        p[i] = algorithm_.Prioritize(
           r_child_node->bound(), *r_child_stat,
           q_child->bound(), *q_stat, *q_result_stat,
           global_stat_);

        r_tree_a_.StopRead(r_child_node, r_child_i);
        r_stat_a_.StopRead(r_child_stat, r_child_i);
      }

      // ensure sorted in descending order
      if (p[0] > p[1]) {
        index_t t;
        r_child_order[0] = r_child_order[1];
        r_child_order[1] = t;
      }

      for (index_t i = 0; i < cardinality; i++) {
        DualTree(q_node_i, r_child_order[i]);
      }
    }
  }

  r_tree_a_.StopRead(r_node, r_node_i);
  r_stat_a_.StopRead(r_stat, r_node_i);
  q_node_a_.StopRead(q_node, q_node_i);
  q_stat_a_.StopRead(q_stat, q_node_i);
  q_result_stat_a_.StopWrite(q_result_stat, q_node_i);
  q_delta_a_.StopWrite(q_delta, q_node_i);
}

/**
 * (Empty) Statistic that can be computed bottom-up.
 *
 * (Use this as a starting point for your own work, except yours may not
 * need to be so templated.)
 */
template<typename TPoint>
class EmptyStat {
 public:
  // Necessities:
  // - Accumulating sub-stats
  // - Weighting (for things like means)
  // - Taking into account my OWN bound

  /** Initialize to a primal value. */
  void StartUpdate() {
  }

  /** Incorporate a single point result. */
  void Update(const TPoint& point, const TInfo& info) {
  }

  /** Incorporate a summary over a sub-result. */
  void Update(const EmptyStat& sub_stat, index_t sub_stat_n_points) {
  }

  /** Do any post-processing necessary for an update step. */
  void StopUpdate(const TBound& bound, index_t n_points) {
  }
};


/**
 * (Empty) Per-point result.
 *
 * This considers results of tuple computations and has no concept
 * of bottom-up accumulation.
 *
 * (Use this as a starting point for your own work, except yours may not
 * need to be so templated.)
 */
template<typename TPoint>
struct EmptyResult{
 public:
  /**
   * Initialize to a sensible initial value before any results have been
   * considered.
   */
  void Init() {
  }

  /**
   * Incorporate a new point into consideration.
   */
  void Update(const TPoint &q, const TPoint& r, index_t r_index) {
  }
};

/**
 * (Empty) Statistic that involves results of tuple-wise computation, that is
 * continually *recomputed* bottom-up.
 *
 * Unlike a bottom-up statistic, this considers the result of computations
 * rather than a summary of points.  This is also constrained to be
 * solely commutative and associative operators and don't take into account
 * bounds themselves.
 *
 * (Use this as a starting point for your own work, except yours may not
 * need to be so templated.)
 */
template<typename TResult>
class EmptyResultStat {
 public:
  // This might need pairs of stats?

  /** Sets to an initial value. */
  void Init() {
  }

  /** Resets value in preparation to be updated. */
  void StartUpdate() {
  }

  /** Incorporates a single result. */
  void Update(const TResult& point_result) {
  }

  /** Incorporates a sub-result. */
  void Update(const EmptyStat& sub_stat) {
  }

  void StopUpdate() {
  }
};


/**
 * (Empty) Change to sub-results for an entire region that is pushed down.
 *
 * This is useful for series expansions for example, where the Delta
 * may store the different moments of the reference points.  These moments
 * can be propagated down in a straightforward manner, and eventually be
 * applied to leaves.
 *
 * (Use this as a starting point for your own work, except yours may not
 * need to be so templated.)
 */
template<typename TResult>
struct EmptyDelta {
 public:
  void Init() {
  }

  void Reset() {
  }

  void ApplyTo(TResultStat *result_stat)  const {
  }

  void ApplyTo(EmptyDelta *sub_delta) const {
  }

  void ApplyTo(const TPoint& point,
      const TInfo& info, TResult *result) const {
  }
};

#endif
