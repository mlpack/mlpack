/**
 * @file gnp.h
 *
 * Framework for defining and executing generalized n-body problems, in
 * serial or parallel.
 */

#ifndef SUPERPAR_GNP_H
#define SUPERPAR_GNP_H


// BIG BIG TODO:
//  - Per reference node WEIGHT is needed for matrix vector multiplication

template<
  typename TAlgorithm,
  typename TPoint, typename TBound,
  typename TRStat,
  typename TQStat, typename QMutStat, typename TMassResult,
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
  /** Stat that is pre-computed for each query node. */
  typedef TQStat QStat;
  /** Stat that is pre-computed for each reference node. */
  typedef TRStat RStat;
  /** Stat updated for each query node as part of the GNP computation. */
  typedef TQMutStat QMutStat;
  /** Stat updated that needs to be distributed to all leaves eventually. */
  typedef TQMassResult QMassResult;
  /** A desired result for each query point. */
  typedef TQResult QResult;
  
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

  /** The dual-tree parts of the algorithm. */
  typedef typename GNP::Algorithm Algorithm;
  /** The type of point considered. */
  typedef typename GNP::Point Point;
  /** The bounding type used for pruning and tree construction. */
  typedef typename GNP::Bound Bound;
  /** Stat that is pre-computed for each query node. */
  typedef typename GNP::QStat QStat;
  /** Stat that is pre-computed for each reference node. */
  typedef typename GNP::RStat RStat;
  /** Stat updated for each query node as part of the GNP computation. */
  typedef typename GNP::QMutStat QMutStat;
  /** Stat updated that needs to be distributed to all leaves eventually. */
  typedef typename GNP::QMassResult QMassResult;
  /** A desired result for each query point. */
  typedef typename GNP::QResult QResult;
  /** Stat computed for the entire computation. */
  typedef typename GNP::GlobalStat GlobalStat;
  
  /** Tree nodes. */
  typedef SpNode< Bound > TreeNode;
 
 private:
  Algorithm algorithm_;
  
  Array<TreeNode> r_tree_a_;
  Array<Point> r_point_a_;
  Array<RStat> r_stat_a_;
  
  Array<TreeNode> q_tree_a_;
  Array<Point> q_point_a_;
  Array<QStat> q_stat_a_;
  Array<QMutStat> q_mut_stat_a_;
  Array<QMassResult> q_mass_result_a_;
  
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
      QMutStat *q_mut_stat) {
  void DualTree_(index_t q_node_i, index_t r_node_i);
  
  void DistributeMassResults_();
  
  void RecursivelyDistributeMassResults_(const TreeNode *q_node,
      QMassResult *q_mass_result);
};


template<class TGNP, class TArray>
void GnpDualTreeRunner<TGNP, TArray>::DistributeMassResults_() {
  if (have_mass_result()) {
    const TreeNode *q_root_node = q_tree_a_.StartRead(
        TreeNode::ROOT_INDEX);
    QMassResult *q_root_mass_result = q_mass_result_a_.StartWrite(
        TreeNode::ROOT_INDEX);
    
    RecursivelyDistributeMassResults_(q_root_node, q_root_mass_result);
  }
}

template<class TGNP, class TArray>
void GnpDualTreeRunner<TGNP, TArray>::RecursivelyDistributeMassResults_(
    const TreeNode *q_node, QMassResult *q_mass_result) {
  q_mass_result->StartDistribute();
  
  for (index_t i = 0; i < q_node->cardinality(); i++) {
    index_t q_child_i = 0;
    const TreeNode *q_child_node = q_tree_a_.StartRead(q_child_i);
    QMassResult *q_child_mass_result = q_mass_result_a_.StartWrite(q_child_i);
    
    q_mass_result->Apply(q_child_mass_result);
    RecursivelyDistributeMassResults_(q_child_node, q_child_mass_result);
    
    q_tree_a_.StopRead(q_child_node, q_child_i);
    q_mass_result_a_.StopWrite(q_child_mass_result, q_child_i);
  }

  q_mass_result->StopDistribute();
}

template<class TGNP, class TArray>
void GnpDualTreeRunner<TGNP, TArray>::BaseCase_(
    const TreeNode *r_node,
    const RStat& r_stat,
    const TreeNode *q_node,
    const QStat& q_stat,
    QMutStat *q_mut_stat) {
  /* base case */
  index_t q_begin = q_node->begin();
  index_t r_begin = r_node->begin();
  index_t q_point_i = q_node->count() + q_begin;
  index_t r_end = r_node->count() + r_begin;

  q_mut_stat->StartUpdate();

  do {
    index_t r_point_i = r_count + r_begin;

    q_point_i--;

    const Point *q_point = q_point_a_.StartRead(q_point_i);
    Result *q_result = q_result_a_.StartWrite(q_point_i);
    // Copy the result to allow it to be register-allocated
    Result q_result_cached(q_result);

    bool did_a_prune = algorithm_.TryPrune(
      r_node->bound(), r_stat,
      q_node->bound(), q_stat, q_mut_stat, q_result,
      &g_stat);
      
    do {
      r_point_i--;
      // INNERMOST LOOP - PERFORMANCE CRUNCH HERE!
      // We might want fast sub-arrays of some sort
      const Point *r_point = r_point_a_.StartRead(r_point_i);
      q_result_cached.Update(*q_point, *r_point);
      r_point_a_.StopWrite(r_point, r_point_i);
    } while (likely(r_point_i > q_begin));

    *q_result = q_result_cached;
    q_result_a_.StopWrite(q_result, q_point_i);
    q_point_a_.StopRead(q_point, q_point_i);

    q_mut_stat->Update(q_result_cached);
  } while (likely(q_point_i > q_begin));

  q_mut_stat->StopUpdate();
}

template<class TGNP, class TArray>
void GnpDualTreeRunner<TGNP, TArray>::DualTree_(
    index_t q_node_i, index_t r_node_i) {
  // Read all data
  const TreeNode *r_node = r_tree_a_.StartRead(r_node_i);
  const RStat *r_stat = r_stat_a_.StartRead(r_node_i);

  const TreeNode *q_node = q_tree_a_.StartRead(q_node_i);
  const QStat *q_stat = q_stat_a_.StartRead(q_node_i);
  QMutStat *q_mut_stat = q_mut_stat_a_.StartWrite(q_node_i);
  QMassResult *q_mass_result = q_mass_result_a_.StartWrite(q_node_i);

  bool did_a_prune = algorithm_.TryPrune(
      r_node->bound(), *r_stat,
      q_node->bound(), *q_stat, q_mut_stat, q_mass_result,
      &global_stat_);
  
  if (!did_a_prune) {
    if (q_node->is_leaf() && r_node->is_leaf()) {
      BaseCase(r_node, r_stat, q_node, q_stat, q_mut_stat);
    } else if ((q_node->count() >= r_node->count() && !q_node->is_leaf())
        || (r_node->is_leaf())) {
      q_mut_stat->StartUpdate();
      for (index_t i = 0; i < q_node->cardinality(); i++) {
        index_t q_child_i = q_node->child(i);
        GnpQueryReferenceRunner::DualTree(q_child_i, r_node_i);
        const QMutStat *q_child_mut_stat = q_mut_stat_a_.StartRead(q_child_i);
        q_mut_stat->Update(q_child_mut_stat);
        q_mut_stat_a_.StopRead(q_child_mut_stat, q_child_i);
      }
      q_mut_stat->StopUpdate();
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
           q_child->bound(), *q_stat, *q_mut_stat,
           global_stat_);
        
        r_tree_a_.StopRead(r_child_node, r_child_i);
        r_stat_a_.StopRead(r_child_stat, r_child_i);
      }
      
      
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
  q_mut_stat_a_.StopWrite(q_mut_stat, q_node_i);
  q_mass_result_a_.StopWrite(q_mass_result, q_node_i);
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
  void StartUpdate() {
  }

  void Update(const TPoint& point) {
  }
  
  void Update(const EmptyStat& sub_stat, index_t sub_stat_n_points) {
  }
  
  void StopUpdate(index_t n_points) {
  }
};

/**
 * (Empty) Statistic that involves results of tuple-wise computation, that is
 * continually *recomputed* bottom-up.
 *
 * The only difference between this and a bottom-up statistic (from a
 * a mechanical standpoint) is that this needs to be able to take on
 * a sane value if no tuples have been considered.
 *
 * (Use this as a starting point for your own work, except yours may not
 * need to be so templated.)
 */
template<typename TResult>
class EmptyMutStat {
 public:
  /** Sets to an initial value. */
  void Init() {
  }
  
  /** Resets value in preparation to be updated. */
  void StartUpdate() {
  }

  void Update(const TResult& point_result) {
  }
  
  void Update(const EmptyStat& sub_stat, index_t sub_stat_n_points) {
  }
  
  void StopUpdate(index_t n_points) {
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
  void Init() {
  }
  
  void Update(const Point &q, const Point& r) {
  }
};

/**
 * (Empty) Sub-result for an entire region that is pushed down.
 *
 * This is useful for series expansions for example, where the MassResult
 * may store the different moments of the reference points.  These moments
 * can be propagated down in a straightforward manner, and eventually be
 * applied to leaves.
 *
 * (Use this as a starting point for your own work, except yours may not
 * need to be so templated.)
 */
template<typename TResult>
struct EmptyMassResult {
 public:
  void Init() {
  }

  //  // I'm deciding between two formulations of this
  //#ifdef UPDATE_VERSION_OF_MASS_RESULTS
  //  // Pro: Looks like the other types of statistics
  //  
  //  // Con: This no longer looks like it's top-down results being "pushed"
  //  // and would instead look a lot more pull-like.
  //  
  //  // Pro: Individual results responsible for everything they need to be
  //  // computed
  //  // Con: When no mass result is needed
  //  // Con: The push-down formation feels like the others, because a parent
  //  // is once again responsible for being called on its children (not children
  //  // responsible for assembling parents' information)
  //  
  //  void Update(const TMassResult& parent_result) {
  //  }
  //  /* Result classes need to be able to assemble results */
  //  
  //#else

  void StartDistribute() {
  }

  void Apply(EmptyMassResult *subresult) const {
  }

  void Apply(TResult *result) const {
  }

  void StopDistribute() {
  }
#endif
};

#endif
