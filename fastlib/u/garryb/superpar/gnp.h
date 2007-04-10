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
class GnpQueryReference {
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
  class GNP, //< A query-reference generalized N-body problem
  class TArray, //< The array type to use for the implementation
>
class GnpQueryReferenceRunner {
 public:
  /** The array type used to store data. */
  typedef TArray Array;

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
  
 private:
  Array< SpNode<Bound> > r_tree_;
  Array<RStat> r_stat_;
  
  Array< SpNode<Bound> > q_tree_;
  Array<QStat> q_stat_;
  Array<QMutStat> q_mut_stat_;
  Array<QMassResult> q_mass_result_;
  
  Array<QResult> q_result_;
  
  GlobalStat global_stat_;
  
 public:
  
 private:
  // TODO: Th
  void DualTree(index_t q_index, index_t r_index); {
    const SpNode<Bound> *r_node = r_tree_.StartRead(r_index);
    const RStat *r_stat = r_bound_.StartRead(r_index);
    
    const SpNode<Bound> *q_node = q_tree_.StartRead(q_index);
    const QStat *q_stat = q_stat_.StartRead(q_index);
    QMutStat *q_mut_stat = q_mut_stat_.StartWrite(q_index);
    QMutStat *q_mass_result = q_mut_stat_.StartWrite(q_index);
    
    if (Algorithm::TryPrune(
        r_node->bound(), *r_stat,
        q_node->bound(), *q_stat, q_mut_stat, q_mass_result,
        &global_stat_) {
      return;
    }
    
    if (q_node.is_leaf() && r_node.is_leaf()) {
      BaseCase(...);
    } else if (q_node.count() >= r_node.count()) {
      
    }
  }
};

template<class GNP, class TArray>
void GnpQueryReferenceRunner::DualTree(index_t q_index, index_t r_index); {
  const SpNode<Bound> *r_node = r_tree_.StartRead(r_index);
  const RStat *r_stat = r_bound_.StartRead(r_index);
  
  const SpNode<Bound> *q_node = q_tree_.StartRead(q_index);
  const QStat *q_stat = q_stat_.StartRead(q_index);
  QMutStat *q_mut_stat = q_mut_stat_.StartWrite(q_index);
  QMutStat *q_mass_result = q_mut_stat_.StartWrite(q_index);
  
  if (Algorithm::TryPrune(
      r_node->bound(), *r_stat,
      q_node->bound(), *q_stat, q_mut_stat, q_mass_result,
      &global_stat_) {
    return;
  }
  
  if (q_node.is_leaf() && r_node.is_leaf()) {
    BaseCase(...);
  } else if ((q_node.count() >= r_node.count() && !q_node.is_leaf())
      || (r_node.is_leaf())) {
    do both q children
    run mut stat accumulator
  } else {
    prioritize references
    run references
  }
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
  
  void FinishUpdate(index_t n_points) {
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
  
  void FinishUpdate(index_t n_points) {
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
  
  void Update(const Point &p) {
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

  void FinishDistribute() {
  }
#endif
};

#endif
