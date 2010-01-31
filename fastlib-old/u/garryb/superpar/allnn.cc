/*

QUESTIONS:
  - when/how are deltas computed?
  - when/how is global stuff updated?
  - can we call "CanPrune" multiple times?  i think it is always best to call
    CanPrune as deep as possible to avoid leaf computations

DELTA SOLUTION?
  - delta is transient
  - MassResult once again stores mass-deltas if it needs to
    - downside: non-lazy version still is space wasteful
    - upside: problems where delta need not be stored explicity (in allnn
    the parent value can substitute)
    - upside: unifies global and local results
  - GlobalResult now cares about deltas
  - still to resolve: does pruning happen before or after deltas are applied?
    - after: deltas must ALWAYS be applied AND undone
    - before: delta does not need to be re-applied within prune function
    - both versions?

MORE PROBLEMS
  - PointResult.AccumulatePair and GlobalResult.AccumulatePair may want to
  share code
  - SOLUTION?
    - Functions no longer separated in weird ways
    - Parameter passing
    - extra parameter to PointResult::AccumulatePair
  - bring back lambda to allow things stored in register?
    - later
  - iterator design pattern?  or allow SIMD?

stat = EmptyStat
*/

struct AllnnStat {
  void Init(const AllnnParam& param) {
  }
  
  void Accumulate(const AllnnParam& param, const Point& point) {
  }
  
  void Accumulate(const AllnnParam& param,
      const AllnnStat& stat, const Bound& bound, index_t n) {
  }
  
  void Finish(const AllnnParam& param, const Bound& bound, index_t n) {
  }
};

struct AllnnPointPairVisitor {
  double best_distance;
  index_t best_index;
  
  void Init(const AllnnParam& param) {
  }
  
  bool StartVisitingQueryPoint(const AllnnParam& param,
      const Point& q_point,
      const RNode& r_node,
      QResult* q_result,
      GlobalResult* global_result) {
    if (r_node.bound().MinDistanceSqToPoint(q_point) > best_distance) {
      return true;
    } else {
      best_distance = q_result->best_distance;
      best_index = q_result->best_index;
      return false;
    }
  }
  
  void VisitPair(const AllnnParam& param,
      const TPoint& q_point, const TQInfo& q_info,
      const TPoint& r_point, const TRInfo& r_info, index_t r_index) {
    double distance = la::DistanceEuclidean(q_point, r_point);
    if (unlikely(distance <= min_distance)) {
      best_distance = distance;
      best_index = r_index;
    }
  }
  
  void FinishVisitingQueryPoint(const AllnnParam& param,
      const TPoint& q_point,
      const TBound& r_bound, const TRStat& r_stat, index_t r_count,
      TResult* q_result,
      TGlobalResult* global_result) {
    q_result->best_distance = best_distance;
    q_result->best_index = best_index;
  }
};

// rho, but a bit of phi and lambda
struct AllnnResult {
  double best_distance;
  index_t best_index;
  
  void Init(const AllnnParam& param,
      const Point& q_point, const QInfo& q_info,
      const RNode& r_root) {
    best_distance = DBL_MAX;
    best_index = -1;
  }
  
  void Finish(const AllnnParam& param,
      const Point& q_point, const QInfo& q_info,
      const RNode& r_root) {
  }
  
  void PullParentResults(const AllnnParam& param,
      const Point& q_point,
      const TMassResult& parent_mass_result) {
    if (unlikely(parent_mass_result < best_distance)) {
      best_distance = parent_mass_result;
      DEBUG_ONLY(best_index = -1);
    }
  }
};

// mu
struct AllnnMassResult {
  double best_distance_u;

  void Init(const AllnnParam& param,
      const QNode& q_node,
      const RNode& r_root) {
    best_distance_u = DBL_MAX;
  }

  void StartReaccumulate(const AllnnParam& param,
      const QNode& q_node) {
    best_distance_u = -DBL_MAX;
  }

  void Accumulate(const AllnnParam& param, const AllnnResult& result) {
    best_distance_u = max(best_distance_u, result.best_distance);
  }

  void Accumulate(const AllnnParam& param,
      const AllnnMassResult& result, index_t n_points) {
    best_distance_u = max(best_distance_u, result.best_distance_u);
  }

  void FinishReaccumulate(const AllnnParam& param,
      const QNode& q_node) {
  }


  void ResetLazyResults(const AllnnParam& param) {
    /* no lazy results to reset */
  }

  bool PullParentResult(const AllnnParam& param,
      const TMassResult& parent_mass_result,
      const QNode& q_node) {
    if (parent_mass_result.best_distance_u < best_distance_u) {
      best_distance_u = parent_mass_result.best_distance_u;
      return CHANGE;
    } else {
      return NO_CHANGE;
    }
  }

  bool ApplyDelta(const AllnnParam& param,
      const Delta& delta, const QNode& q_node) {
    if (delta.distance_u < best_distance_u) {
      best_distance_u = delta.distance_u;
      return CHANGE;
    } else {
      return NO_CHANGE;
    }
  }

  void UndoDelta(const AllnnParam& param,
      const Delta& delta, const QNode& q_node) {
    /* no undo operation necessary */
  }

  void Finish(const AllnnParam& param) {
  }
};

// delta
class AllnnDelta {
  double distance_u;
  
  void Init(const AllnnParam& param) {
  }
  
  void Compute(const AllnnParam& param,
      const QNode& q_node,
      const RNode& r_node,
      const TMassResult& q_mass_result) {
    distance_u = q_node.bound().MaxDistanceToBound(r_node.bound());
  }
};

// DeltaUndoInformation

class AllnnGlobalResult {
  void Init(const AllnnParam& param) {
  }
  
  void Accumulate(const AllnnParam& param,
      const AllnnGlobalResult& other_global_result) {
  }
  
  void ApplyDelta(const AllnnParam& param,
      const AllnnDelta& delta) {
  }
  
  void UndoDelta(const AllnnParam& param,
      const AllnnDelta& delta) {
  }
  
  void Finish(const AllnnParam& param) {
  }
};

class AllnnAlgorithm {
  void Init(const AllnnParam& param) {
  }
  
  /**
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
  static bool MustExplore(
      const AllnnParam& param,
      const QNode& q_node,
      const RNode& r_node,
      const TDelta& delta,
      const TMassResult& q_mass_result, const TGlobalResult& global_result) {
    double distance_l = q_node.bound().MinDistanceSqToBound(r_node.bound());
    return distance_l <= q_mass_result.best_distance_u;
  }
  
  /**
   * Computes a heuristic for how early a computation should occur -- smaller
   * values are earlier.
   */
  static double Heuristic(
      const AllnnParam& param,
      const QNode& q_node,
      const RNode& r_node,
      const TMassResult& q_mass_result) {
    return q_node.bound().MidDistanceSqToBound(r_node.bound());
  }
};
