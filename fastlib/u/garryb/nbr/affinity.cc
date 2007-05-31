
struct AffinityCommon {
  /** The bounding type. Required by NBR. */
  typedef SpHrectBound<2> Bound;

  struct Alpha {
   public:
    double max1;
    double max2;
    index_t max1_index;

    OT_DEF(Alpha) {
      OT_MY_OBJECT(max1);
      OT_MY_OBJECT(max2);
      OT_MY_OBJECT(max1_index);
    }

   public:
    double get(index_t i) {
      if (unlikely(i == max1_index)) {
        return max2;
      } else {
        return max1;
      }
    }
  };

  struct AlphaInfo {
    Alpha alpha;

    OT_DEF(AlphaInfo) {
      OT_MY_OBJECT(alpha);
    }
  };
  
  struct RhoInfo {
    double rho;

    OT_DEF(RhoInfo) {
      OT_MY_OBJECT(rho);
    }
  };
  
  typedef SpVectorPoint<AlphaInfo> AlphaPoint;
  typedef SpVectorPoint<RhoInfo> RhoPoint;

  struct Param {
   public:
    /** The epsilon for approximation. */
    double eps;
    /** The dimensionality of the data sets. */
    index_t dim;
    /** Number of points */
    index_t n_points;

    OT_DEF(Param) {
      OT_MY_OBJECT(eps);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(n_points);
    }

   public:
    void Copy(const Param& other) {
      eps = other.eps;
      dim = other.dim;
    }

    void Init(datanode *module) {
      dim = -1;
      eps = fx_param_double(module, "eps", 1.0e-2);
    }

    void AnalyzePoint(const AlphaPoint& q) {
      if (dim == -1) {
        dim = q.vec().length();
      } else {
        DEBUG_ASSERT_MSG(dim == q.length(), "Differing dimensionality");
      }
    }
    void AnalyzePoint(const RhoPoint& r) {
      if (dim == -1) {
        dim = q.vec().length();
      } else {
        DEBUG_ASSERT_MSG(dim == q.length(), "Differing dimensionality");
      }
    }

   public:
    double Similarity(const Vector& q, const Vector& r) const {
      return 1.0 / sqrt(la::EuclideanDistanceSq(a, b));
    }
    double Similarity(
        const Vector& q, index_t q_index,
        const Vector& r, index_t r_index) const {
      if (unlikely(q_index == r_index)) {
        return pref;
      }
      return Similarity(q, r);
    }
    double SimilarityHi(const AlphaNode& a, const AlphaNode& b) const {
      double dist = sqrt(a->bound().MinDistanceSqToBound(b->bound()));
      double hi = 1.0 / dist;
      if (q->begin() < r->end() && r->begin() < q->end() && pref > upper_bound) {
        hi = pref;
      }
      return hi;
    }
    
    double ErrorShare(double abs_error_used, const RNode& r_node) {
      return (eps - abs_error_used) * r_node.count() / n_points;
    }
  };


  struct AlphaStat {
   public:
    SpRange alpha;
    
    OT_DEF(AlphaStat) {
      OT_MY_OBJECT(alpha);
    }

   public:
    void Init(const Param& param) {
      alpha.InitEmptySet();
    }
    void Accumulate(const Param& param, const AlphaPoint& point) {
      alpha |= SpRange(point.info().max2, point.info().max1);
    }
    void Accumulate(const Param& param,
        const AlphaStat& stat, const Bound& bound, index_t n) {
      alpha |= stat.alpha;
    }
    void Postprocess(const Param& param, const Bound& bound, index_t n) {}
  };

  struct RhoStat {
   public:
    SpRange rho;
    
    OT_DEF(RhoStat) {
      OT_MY_OBJECT(rho);
    }
    
   public:
    void Init(const Param& param) {
      rhos.InitEmptySet();
    }
    void Accumulate(const Param& param, const RhoPoint& point) {
      rhos |= SpRange(point.info().rho);
    }
    void Accumulate(const Param& param,
        const RhoStat& stat, const Bound& bound, index_t n) {
      rhos |= stat.rhos;
    }
    void Postprocess(const Param& param, const Bound& bound, index_t n) {}
  };

  typedef SpNode<Bound, AlphaStat> AlphaNode;
  typedef SpNode<Bound, RhoStat> RhoNode;
};

class AffinityAlpha {
 public:
  typedef AffinityCommon::AlphaPoint QPoint;
  typedef AffinityCommon::RhoPoint RPoint;

  typedef AffinityCommon::Param Param;

  typedef AffinityCommon::AlphaStat QStat;
  typedef AffinityCommon::RhoStat RStat;

  typedef AffinityCommon::AlphaNode QNode;
  typedef AffinityCommon::RhoNode RNode;

  typedef BlankGlobalResult GlobalResult;

  struct BlankPostponed QPostponed;

  struct Delta {
   public:
    SpRange alpha;

    OT_DEF(Delta) {
      OT_MY_OBJECT(alpha);
    }

   public:
    void Init(const Param& param) {
    }
  };

  struct QResult {
   public:
    Alpha alpha;
    
    OT_DEF(QResult) {
      OT_MY_OBJECT(alpha);
    }

   public:
    void Init(const Param& param) {
      alpha.max1 = -DBL_MAX;
      alpha.max2 = -DBL_MAX;
      alpha.max1_index = -1;
    }
    void Postprocess(const Param& param,
        const QPoint& q, const RNode& r_root) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QPoint& q) {}
  };

  struct QMassResult {
   public:
    SpRange alpha;

    OT_DEF(QMassResult) {
      OT_MY_OBJECT(alpha_hi);
      OT_MY_OBJECT(alpha_lo);
      OT_MY_OBJECT(xyz);
    }

   public:
    void Init(const Param& param) {
      alpha.InitUniversalSet();
    }
    void ApplyMassResult(const Param& param, const QMassResult& mass_result) {
      alpha.MaxWith(mass_result.alpha);
    }
    void ApplyDelta(const Param& param, const Delta& delta) {
      alpha.MaxWith(delta.alpha);
    }
    bool ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {}
    void StartReaccumulate(const Param& param, const QNode& q_node) {
      alpha.InitEmptySet();
    }
    void Accumulate(const Param& param, const QResult& result) {
      alpha |= SpRange(result.max2, result.max1);
    }
    void Accumulate(const Param& param,
        const QMassResult& result, index_t n_points) {
      alpha |= result.alpha;
    }
    void FinishReaccumulate(const Param& param, const QNode& q_node) {}
  };

  struct PairVisitor {
   public:
    Alpha alpha;
    
   public:
    void Init(const Param& param) {}

    bool StartVisitingQueryPoint(const Param& param,
        const Point& q,
        const RNode& r_node, const QMassResult& unapplied_mass_results,
        QResult* q_result, GlobalResult* global_result) {
      alpha = q_result->alpha;
    }
    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
      double cur_alpha;
      
      if (unlikely(q_index == r_index)) {
        cur_alpha = r.info().rho + q.info().alpha.get(r_index);
      } else {
        double sim = param.Similarity(q.vec(), r.vec());
        cur_alpha = min(
            min(sim, q.info().alpha.get(r_index)) + r.info().rho,
            sim);
      }
      
      if (unlikely(cur_alpha > alpha.max2)) {
        if (unlikely(cur_alpha > alpha.max1)) {
          alpha.max2 = alpha.max1;
          alpha.max1 = cur_alpha;
          alpha.max_index = r_index;
        } else {
          alpha.max2 = cur_alpha;
        }
      }
    }
    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q,
        const RNode& r_node, const QMassResult& unapplied_mass_results,
        QResult* q_result, GlobalResult* global_result) {
      q_result->alpha = alpha;
    }
  };

  class Algorithm {
   public:
    static bool ConsiderPairIntrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node,
        Delta* delta,
        GlobalResult* global_result, QPostponed* q_postponed) {
      double sim_hi = param.SimilarityHi(q_node, r_node);
      double sim_lo = param.SimilarityLo(q_node, r_node);

      delta->alpha.lo = min(
          min(q_node.stat().alpha.lo, sim_lo) + r_node.stat().rho.lo,
          sim_lo);
      delta->alpha.hi =
          q_node.stat().alpha.hi + r_node.stat().rho.hi;

      return true;
    }
    static bool ConsiderPairExtrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      if (delta.alpha.hi <= q_mass_result.alpha.lo) {
        return false;
      } else {
        return true;
      }
    }
    static bool ConsiderQueryTermination(const Param& param,
        const QNode& q_node,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
    }
    static double Heuristic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta) {
      return -delta.sim_hi;
    }
  };
};

class AffinityRho {
 public:
  typedef AffinityCommon::Alpha Alpha;

  typedef AffinityCommon::RhoPoint QPoint;
  typedef AffinityCommon::AlphaPoint RPoint;

  typedef AffinityCommon::Param Param;

  typedef AffinityCommon::RhoStat QStat;
  typedef AffinityCommon::AlphaStat RStat;

  typedef AffinityCommon::AlphaNode QNode;
  typedef AffinityCommon::RhoNode RNode;

  typedef BlankGlobalResult GlobalResult;

  struct QPostponed {
   public:
    double d_rho;
    double abs_error_used;
    
    OT_DEF(QPostponed) {
      OT_MY_OBJECT(d_rho);
      OT_MY_OBJECT(abs_error_used);
    }
    
   public:
    void Init(const Param& param) {
      Reset();
    }

    void Reset(const Param& param) {
      d_rho = 0;
      abs_error_used = 0;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      d_rho += other.d_rho;
      abs_error_used += other.abs_error_used;
    }
  };

  struct Delta {
   public:
    SpRange d_rho;

    OT_DEF(Delta) {
      OT_MY_OBJECT(d_rho);
    }

   public:
    void Init(const Param& param) {
    }
  };

  struct QResult {
   public:
    double rho;
    double abs_error_used;
    
    OT_DEF(QResult) {
      OT_MY_OBJECT(rho);
      OT_MY_OBJECT(abs_error_used);
    }

   public:
    void Init(const Param& param) {
      rho = 0;
      abs_error_used = 0;
    }
    void Postprocess(const Param& param,
        const QPoint& q, const RNode& r_root) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QPoint& q) {
      rho += postponed.d_rho;
      abs_error_used += postponed.abs_error_used;
    }
  };

  struct QMassResult {
   public:
    SpRange rho;
    double abs_error_used;

    OT_DEF(QMassResult) {
      OT_MY_OBJECT(rho);
      OT_MY_OBJECT(abs_error_used);
    }

   public:
    void Init(const Param& param) {
      rho.Init(0, 0);
      abs_error_used = 0;
    }
    void ApplyMassResult(const Param& param, const QMassResult& mass_result) {
      rho += mass_result.rho;
      abs_error_used += mass_result.abs_error_used;
    }
    void ApplyDelta(const Param& param, const Delta& delta) {
      rho += delta.d_rho;
    }
    bool ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      rho += postponed.d_rho;
      abs_error_used += postponed.abs_error_used;
    }
    void StartReaccumulate(const Param& param, const QNode& q_node) {
      rho.InitEmptySet();
      abs_error_used = 0;
    }
    void Accumulate(const Param& param, const QResult& result) {
      rho |= result.rho;
      abs_error_used = max(abs_error_used, result.abs_error_used);
    }
    void Accumulate(const Param& param,
        const QMassResult& result, index_t n_points) {
      rho |= result.rho;
      abs_error_used = max(abs_error_used, result.abs_error_used);
    }
    void FinishReaccumulate(const Param& param, const QNode& q_node) {}
  };

  struct PairVisitor {
   public:
    double rho;

   public:
    void Init(const Param& param) {}
    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q,
        const RNode& r_node, const QMassResult& unapplied_mass_results,
        QResult* q_result, GlobalResult* global_result) {
      rho = q_result->rho;
    }
    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
      double sim = param.Similarity(q.vec(), r.vec())
          - r.info().alpha.get(r_index);

      if (sim < 0 && likely(q_index != r_index)) {
        sim = 0;
      }
      
      rho += sim;
    }
    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q,
        const RNode& r_node, const QMassResult& unapplied_mass_results,
        QResult* q_result, GlobalResult* global_result) {
      q_result->rho = rho;
    }
  };

  class Algorithm {
   public:
    static bool ConsiderPairIntrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node,
        Delta* delta,
        GlobalResult* global_result, QPostponed* q_postponed) {
      double sim_hi = param.SimilarityHi(q_node, r_node);
      double sim_lo = param.SimilarityLo(q_node, r_node);

      delta->rho.lo = (sim_lo - r_node.stat().alpha.hi) * r_node.count();
      delta->rho.hi = max(0, sim_hi - r_node.stat().alpha.lo) * r_node.count();

      return true;
    }
    static bool ConsiderPairExtrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      double abs_error = delta->d_rho.width() / 2;
      double rel_error_hi = abs_error / q_mass_result.rho.lo;
      
      if (rel_error_hi < param.ErrorShare(q_mass_result.abs_error_used, r_node)) {
        q_postponed->abs_error_used += abs_error;
        q_postponed->d_rho += delta->d_rho.mid();
        return false;
      } else {
        return true;
      }
    }
    static bool ConsiderQueryTermination(const Param& param,
        const QNode& q_node,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
    }
    static double Heuristic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta) {
      // favor whatever brings our lower bound up the fastest
      return -delta.d_rho_lo;
    }
  };
};


void AffinityMain(datanode *module, const char *gnp_name) {
  typename GNP::Param param;

  param.Init(fx_submodule(module, gnp_name, gnp_name));

  TempCacheArray<typename GNP::QData> q_points;
  TempCacheArray<typename GNP::QNode> q_nodes;
  TempCacheArray<typename GNP::RData> r_points;
  TempCacheArray<typename GNP::RNode> r_nodes;
  TempCacheArray<typename GNP::QResult> q_results;

  nbr_utils::LoadKdTree(fx_submodule(module, "q", "q"),
      &param, &q_points, &q_nodes);
  nbr_utils::LoadKdTree(fx_submodule(module, "r", "r"),
      &param, &r_points, &r_nodes);

  typename GNP::QResult default_result;
  default_result.Init(param);
  q_results.Init(default_result, q_points.end_index(),
      q_points.n_block_elems());

  Solver solver;
  solver.InitSolve(fx_submodule(module, "solver", "solver"), param, 0,
      q_points.cache(), q_nodes.cache(),
      r_points.cache(), r_nodes.cache(), q_results.cache());
}
