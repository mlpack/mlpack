

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

  // rho, but a bit of phi and lambda
  template<typename AllnnParam, typename Point, typename AllnnDelta>
  struct AllnnResult {
    double best_distance;
    index_t best_index;
    
    void Init(const AllnnParam& param,
        const TBound& r_global_bound, const TStat& r_global_stat,
        index_t r_global_count) {
      best_distance = DBL_MAX;
      best_index = -1;
    }
    
    // WALDO: Prototype changed
    void AccumulatePair(const AllnnParam& param,
         const Point& q, const Point& r, index_t r_index,
         GlobalResult *global_result) {
       double d = la::DistanceSqEuclidean(q, r);
       if (unlikely(d <= best_distance)) {
         best_distance = d;
         best_index = r_index;
       }
    }
    
    void Finish(const AllnnParam& param) {
    }
    
    void Apply(const AllnnParam& param, const AllnnDelta& delta, const Point& q) {
      if (unlikely(delta.distance_hi < best_distance)) {
        best_distance = delta.distance_hi;
        DEBUG_ONLY(best_index = -1);
      }
    }
  };

  // mu
  template<typename AllnnParam, typename AllnnResult, typename AllnnDelta>
  struct EmptyMassResult {
    double best_distance_hi;
    
    // must be called with enough information to reproduce its child's value
    void Init(const AllnnParam& param,
        const TBound& q_bound, const TQStat& q_stat, index_t q_count,
        const TBound& r_global_bound, const TRStat& r_global_stat,
        index_t r_global_count) {
      best_distance_hi = DBL_MAX;
    }
    
    void StartReaccumulate(const AllnnParam& param) {
      best_distance_hi = -DBL_MAX;
    }
    
    void Accumulate(const AllnnParam& param, const AllnnResult& result) {
      if (unlikely(result.best_distance > best_distance_hi)) {
        best_distance_hi = result.best_distance;
      }
    }
    
    void Accumulate(const AllnnParam& param,
        const EmptyMassResult& result, index_t n_points) {
      best_distance_hi = max(best_distance_hi, distance_hi);
    }
    
    void Finish(const AllnnParam& param) {
    }
    
    bool Apply(const AllnnParam& param, const AllnnDelta& delta,
        const TStat& stat, const TBound& bound, index_t n) {
      if (unlikely(delta.distance_hi < best_distance_hi)) {
        best_distance_hi = delta.distance_hi;
        return true;
      } else {
        return false;
      }
    }
  };

  // delta
  class EmptyDelta {
    // TODO: An agnostic init function for when it's going to be clobbered?
    double distance_hi;
    
    enum { NEED_UNAPPLY = 0; }
    
    void InitCompute(const AllnnParam& param,
        const AllnnParam& params,
        const Bound& q_bound, const EmptyStat& q_stat, index_t q_count,
        const Bound& r_bound, const EmptyStat& r_stat, index_t r_count) {
      
    }
    
    void Init(const AllnnParam& param) {
      distance_hi = DBL_MAX;
    }
    
    void Reset(const AllnnParam& param) {
      distance_hi = DBL_MAX;
    }
    
    void Apply(const AllnnDelta& delta) {
      distance_hi = max(distance_hi, delta.distance_hi);
    }
    
    void Unapply(const AllnnDelta& delta) {
      // no "reverse application" necessary
    }
  };


  class AllnnAlgorithm {
    void Init(const TParam& param) {
    }
    
    /**
     *
     * delta behavior: delta comes in UNinitialized.  do not make changes
     * to mass result yourself, use delta to do it for you.
     *
     * @param q_bound bound of query node
     * @param q_stat statistic of query node
     * @param q_count number of points in query subtree
     * @param r_bound bound of reference node
     * @param r_stat statistic of reference node
     * @param r_count number of points in query subtree
     * @param delta a delta to update (it begins completely unitialized)
     * @param q_mass_result the mass result to update
     * @param global_result global result to update
     * @return whether a prune occured and further exploration stops
     */
    static bool Consider(
        const AllnnParam& params,
        const Bound& q_bound, const EmptyStat& q_stat, index_t q_count,
        const Bound& r_bound, const EmptyStat& r_stat, index_t r_count,
        const AllnnDelta& delta, AllnnMassResult* q_mass_result,
        EmptyGlobalResult* global_result) {
      double distance_lo = q_bound.MinDistanceTo(r_bound);
      
      return distance_lo > q_mass_result->distance_hi;
    }
    
    /**
     * Computes a heuristic for how early a computation should occur -- smaller
     * values are earlier.
     */
    static double Heuristic(
        const AllnnParam& params,
        const Bound& q_bound, const EmptyStat& q_stat, index_t q_count,
        const Bound& r_bound, const EmptyStat& r_stat, index_t r_count,
        const AllnnDelta& delta,
        const AllnnResult& q_mass_result) {
      return q_bound.MidDistanceTo(r_bound);
    }
  };
