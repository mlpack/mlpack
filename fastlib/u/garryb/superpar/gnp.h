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


// sigma
template<typename TParam, typename TPoint, typename TBound>
struct EmptyStat {
  void Init(const TParam& param) {
  }
  
  void Accumulate(const TParam& param, const TPoint& point) {
  }
  
  void Accumulate(const TParam& param,
      const EmptyStat& stat, const TBound& bound, index_t n) {
  }
  
  void Finish(const TParam& param, const TBound& bound, index_t n) {
  }
};

// rho, but a bit of phi and lambda
template<typename TParam, typename TPoint, typename TDelta>
struct EmptyResult {
  void Init(const TParam& param,
      const TBound& r_global_bound, const TStat& r_global_stat,
      index_t r_global_count) {
  }
  
  void AccumulatePair(const TParam& param,
     const TPoint& q, const TPoint& r) {
  }
  
  void Finish(const TParam& param) {
  }
  
  void Apply(const TParam& param, const TDelta& delta, const TPoint& q) {
  }
};

// mu
template<typename TParam, typename TResult, typename TDelta>
struct EmptyMassResult {
  // must be called with enough information to reproduce its child's value
  void Init(const TParam& param,
      const TBound& q_bound, const TQStat& q_stat, index_t q_count,
      const TBound& r_global_bound, const TRStat& r_global_stat,
      index_t r_global_count) {
  }
  
  void StartReaccumulate(const TParam& param) {
  }
  
  void Accumulate(const TParam& param, const TResult& result) {
  }
  
  void Accumulate(const TParam& param,
      const EmptyMassResult& result, index_t n_points) {
  }
  
  void Finish(const TParam& param) {
  }
  
/*

delta behaviors

                33,0
           /            \
         21,0          12,0
        /   \          /   \
      1,20 0,20      1,10 1,10

  - STRICT breadth first
    - when i consider a node, i:
      - check for prune conditions
      - if i can prune:
        - postpone my delta (which was already applied) so it will apply
        to entire tree
      - if mu has postponed deltas (i.e. a prune occured higher up)
        - apply postponed deltas to q children
        - clear my postponed deltas
        - possible cache problem
      - if i cannot prune:
        - calculate deltas for q children
        - enqueue these
        - if needed, fix my own value: undo my own delta, and apply
        children deltas; this might give better pruning info at current level
  - depth first
    - when i consider a node, i:
      - check if i can prune
      - apply deltas?!?!?!?!
        - for idempotent functions you can apply a delta to aid pruning
        - i don't think this is ever helpful
      - if i can prune:
        - apply and postpone my delta
        - example
          - exact thresholded Epanechnikov KDE inclusion:
            store moments.  conservative updates to density bounds are
            questionable -- they would have to be undone.
      - if i cannot prune:
        - for each q child
          - apply postponed deltas
          - 
        - recalculate mu (this implicitly clears all postponed deltas)

*/

  // void ApplyDeltas(const TParam& param,
  //     const MassResult& foo,
  //     const TStat& stat, const TBound& bound, index_t n) {
  // }  
  // 
  // void PostponeDelta(const TParam& param, const TDelta& delta,
  //     const TStat& stat, const TBound& bound, index_t n) {
  // }
};

// delta
class EmptyDelta {
  // no data necessary
};

// DeltaUndoInformation

// gamma - only knows how to accumulate self, assume other functions smart
class EmptyGlobalResult {
  void Init(const TParam& param) {
  }
  
  void Accumulate(const TParam& param, const EmptyGlobalResult& global_result) {
  }
  
  void Finish(const TParam& param) {
  }
};

class EmptyAlgorithm {
  void Init(const TParam& param) {
  }
  
  /**
   *
   * delta behavior: delta comes in initialized.  do not make changes
   * to mass result yourself, use delta to do it for you.
   *
   * @param q_bound bound of query node
   * @param q_stat statistic of query node
   * @param q_count number of points in query subtree
   * @param r_bound bound of reference node
   * @param r_stat statistic of reference node
   * @param r_count number of points in query subtree
   * @param delta a delta to update (it begins UNinitialized)
   * @param q_mass_result the mass result to update
   * @param global_result global result to update
   * @return whether a prune occured and further exploration stops
   */
  static bool Consider(
      const TParam& params,
      const TBound& q_bound, const TQStat& q_stat, index_t q_count,
      const TBound& r_bound, const TRStat& r_stat, index_t r_count,
      TDelta* delta, TMassResult* q_mass_result, TGlobalResult* global_result) {
  }
  
  /**
   * Computes a heuristic for how early a computation should occur -- smaller
   * values are earlier.
   */
  static double Heuristic(
      const TParam& params,
      const TBound& q_bound, const TQStat& q_stat, index_t q_count,
      const TBound& r_bound, const TRStat& r_stat, index_t r_count,
      const TMassResult& q_mass_result) {
  }
};

#endif
