#include "spbounds.h"
#include "gnp.h"
#include "dfs.h"
#include "nbr_utils.h"

#include "fastlib/fastlib.h"

struct AffinityCommon {
  /** The bounding type. Required by NBR. */
  typedef SpHrectBound<2> Bound;

  /**
   * Alpha corresponds to "maximum availability" with the != k condition.
   */
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
    double get(index_t i) const {
      if (unlikely(i == max1_index)) {
        return max2;
      } else {
        return max1;
      }
    }
  };

  struct CombinedInfo {
    /** Maximum availability of the point. */
    Alpha alpha;
    /** Sum of responsibilities. */
    double rho;

    OT_DEF(CombinedInfo) {
      OT_MY_OBJECT(alpha);
      OT_MY_OBJECT(rho);
    }
  };

  typedef SpVectorInfoPoint<CombinedInfo> CombinedPoint;

  struct Param {
   public:
    /** The epsilon for approximation. */
    double eps;
    /** The dimensionality of the data sets. */
    index_t dim;
    /** Number of points */
    index_t n_points;
    /** Self-pereference. */
    double pref;
    /** The damping factor. */
    double lambda;

    OT_DEF(Param) {
      OT_MY_OBJECT(eps);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(n_points);
      OT_MY_OBJECT(pref);
      OT_MY_OBJECT(lambda);
    }

   public:
    void Copy(const Param& other) {
      pref = other.pref;
      eps = other.eps;
      dim = other.dim;
      n_points = other.n_points;
      lambda = other.lambda;
    }

    void Init(datanode *module) {
      dim = -1;
      eps = fx_param_double(module, "eps", 1.0e-2);
      pref = fx_param_double_req(module, "pref");
      lambda = fx_param_double(module, "lambda", 0.8);
    }

    void BootstrapMonochromatic(CombinedPoint *point, index_t count) {
      dim = point->vec().length();
      n_points = count;
      // TODO: Realistic values
      point->info().rho = 0;
      point->info().alpha.max1 = 0;
      point->info().alpha.max2 = 0;
      point->info().alpha.max1_index = 0;
    }
  };

  struct CombinedStat {
   public:
    SpRange alpha;
    SpRange rho;

    OT_DEF(CombinedStat) {
      OT_MY_OBJECT(alpha);
      OT_MY_OBJECT(rho);
    }

   public:
    void Init(const Param& param) {
      Reset(param);
    }
    void Reset(const Param& param) {
      alpha.InitEmptySet();
      rho.InitEmptySet();
    }
    void Accumulate(const Param& param, const CombinedPoint& point) {
      alpha |= SpRange(point.info().alpha.max2, point.info().alpha.max1);
      rho |= point.info().rho;
    }
    void Accumulate(const Param& param,
        const CombinedStat& stat, const Bound& bound, index_t n) {
      alpha |= stat.alpha;
      rho |= stat.rho;
    }
    void Postprocess(const Param& param, const Bound& bound, index_t n) {}
  };

  typedef SpNode<Bound, CombinedStat> CombinedNode;

  struct Helpers {
    static double Similarity(double distsq) {
      return -distsq;
    }
    static double Similarity(const Vector& a, const Vector& b) {
      //uint32 anum = (mem::PointerAbsoluteAddress(a.ptr()) * 315187727);
      //uint32 bnum = (mem::PointerAbsoluteAddress(b.ptr()) * 210787727);
      //uint32 val = ((anum - bnum) >> 16) & 0xfff;
      //double noise = (1.0e-5 / 4096) * val;
      return Similarity(la::DistanceSqEuclidean(a, b));
    }
    static double Similarity(
        const Param& param,
        const Vector& q, index_t q_index,
        const Vector& r, index_t r_index) {
      if (unlikely(q_index == r_index)) {
        return param.pref;
      } else {
        return Similarity(q, r);
      }
    }
    static double SimilarityHi(
        const Param& param,
        const CombinedNode& a, const CombinedNode& b) {
      double distsq = a.bound().MinDistanceSqToBound(b.bound());
      double hi = Similarity(distsq);
      if (a.begin() < b.end() && b.begin() < a.end()
          && param.pref > hi) {
        hi = param.pref;
      }
      return hi;
    }
    static double SimilarityLo(
        const Param& param,
        const CombinedNode& a, const CombinedNode& b) {
      double distsq = a.bound().MaxDistanceSqToBound(b.bound());
      double lo = Similarity(distsq);
      if (a.begin() < b.end() && b.begin() < a.end()
          && param.pref < lo) {
        lo = param.pref;
      }
      return lo;
    }
    
    static double ErrorShare(const Param& param,
        double abs_error_used, const CombinedNode& r_node) {
      return (param.eps - abs_error_used) * r_node.count() / param.n_points;
    }
  };
};

class AffinityAlpha {
 public:
  typedef AffinityCommon::CombinedPoint QPoint;
  typedef AffinityCommon::CombinedPoint RPoint;

  typedef AffinityCommon::Alpha Alpha;
  
  typedef AffinityCommon::Param Param;

  typedef AffinityCommon::CombinedNode QNode;
  typedef AffinityCommon::CombinedNode RNode;

  typedef BlankGlobalResult GlobalResult;

  typedef BlankQPostponed QPostponed;

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
      alpha.max1 = param.pref;
      alpha.max2 = param.pref;
      alpha.max1_index = -1;
    }
    void Postprocess(const Param& param,
        const QPoint& q, index_t q_index, const RNode& r_root) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QPoint& q) {}
  };

  struct QMassResult {
   public:
    SpRange alpha;

    OT_DEF(QMassResult) {
      OT_MY_OBJECT(alpha);
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
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {}
    void StartReaccumulate(const Param& param, const QNode& q_node) {
      alpha.InitEmptySet();
    }
    void Accumulate(const Param& param, const QResult& result) {
      alpha |= SpRange(result.alpha.max2, result.alpha.max1);
    }
    void Accumulate(const Param& param,
        const QMassResult& result, index_t n_points) {
      alpha |= result.alpha;
    }
    void FinishReaccumulate(const Param& param, const QNode& q_node) {}
  };


  /*
  
  As two-variable functions:
  
  \rho(i, k) = \sum_{j != i, j != k} max(0, S(j,k) - \alpha(j,k))
  
  \alpha(i, k) = min(0, \max_{j != k} S(j,j) + \alpha(j,j) + \rho(i,j))
  
  As one-variable rho:
  
  \rho(k) = \sum_{j != k} max(0, S(j,k) - \alpha(j,k))
  
  \sum max(0, S(j,k) - \alpha(j,k)) - max(0, S(k,k) - alpha(k,k))
  
  \alpha(i, k) = min(0, \max_{j != k} S(j,j) + \alpha(j,j)
       + \rho(i) - max(0, S(i, j) - \alpha(i, j)))

  \alpha(i) = min(0, \max^2{j}
        S(j,j) + \alpha(j,j) + \rho(j) - max(0, S(i, j) - \alpha(i, j)))
   except when i = j in which case we don't need to do the second part

        S(j,j) + \alpha(j,j) + \rho(j) - S(i, j) + min(S(i,j), \alpha(i, j))
  */

  struct PairVisitor {
   public:
    Alpha alpha;
    
   public:
    void Init(const Param& param) {}

    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q,
        const RNode& r_node, const QMassResult& unapplied_mass_results,
        QResult* q_result, GlobalResult* global_result) {
      // We could add a pruning rule here to speed things up quite a bit.
      alpha = q_result->alpha;
      return true;
    }
    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index, const RPoint& r, index_t r_index) {
      double candidate_alpha;

      if (likely(q_index != r_index)) {
        double sim = AffinityCommon::Helpers::Similarity(q.vec(), r.vec());
        candidate_alpha = min(
            min(sim, q.info().alpha.get(r_index)) + r.info().rho,
            sim);
      } else {
        candidate_alpha = r.info().rho + q.info().alpha.get(r_index);
      }

      if (unlikely(candidate_alpha > alpha.max2)) {
        if (unlikely(candidate_alpha > alpha.max1)) {
          alpha.max2 = alpha.max1;
          alpha.max1 = candidate_alpha;
          alpha.max1_index = r_index;
        } else {
          alpha.max2 = candidate_alpha;
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
      //WALDO
      double sim_lo = AffinityCommon::Helpers::SimilarityLo(
          param, q_node, r_node);

      delta->alpha.lo = min(
          min(q_node.stat().alpha.lo, sim_lo) + r_node.stat().rho.lo,
          sim_lo);
      if (q_node.begin() < r_node.end() && r_node.begin() < q_node.end()) {
        delta->alpha.hi =
            q_node.stat().alpha.hi + r_node.stat().rho.hi;
      } else {
        double sim_hi = AffinityCommon::Helpers::SimilarityHi(
            param, q_node, r_node);
        delta->alpha.hi = min(
            min(q_node.stat().alpha.hi, sim_hi) + r_node.stat().rho.hi,
            sim_hi);
      }

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
      return -delta.alpha.hi;
    }
  };
};

class AffinityRho {
 public:
  typedef AffinityCommon::Alpha Alpha;

  typedef AffinityCommon::CombinedPoint QPoint;
  typedef AffinityCommon::CombinedPoint RPoint;

  typedef AffinityCommon::Param Param;

  typedef AffinityCommon::CombinedNode QNode;
  typedef AffinityCommon::CombinedNode RNode;

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
      Reset(param);
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
        const QPoint& q, index_t q_index, const RNode& r_root) {
      double responsibility =
          param.pref - q.info().alpha.get(q_index);

      // Make sure we count ourselves regardless of sign.
      if (responsibility < 0) {
        rho += responsibility;
      }
    }
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
    void ApplyPostponed(const Param& param,
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
      return true;
    }
    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
      double responsibility =
          AffinityCommon::Helpers::Similarity(
              param, q.vec(), q_index, r.vec(), r_index)
          - r.info().alpha.get(r_index);

      if (responsibility > 0) {
        rho += responsibility;
      }
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
      double sim_hi = AffinityCommon::Helpers::SimilarityHi(
          param, q_node, r_node);
      double sim_lo = AffinityCommon::Helpers::SimilarityLo(
          param, q_node, r_node);

      // fprintf(stderr, "(%d,%d) alpha.lo,hi = (%f, %f)\n",
      //           r_node.begin(), r_node.end(),
      //           r_node.stat().alpha.hi,
      //           r_node.stat().alpha.lo
      //           );
      delta->d_rho.lo = max(0.0, sim_lo - r_node.stat().alpha.hi)
          * r_node.count();
      delta->d_rho.hi = max(0.0, sim_hi - r_node.stat().alpha.lo)
          * r_node.count();

      return delta->d_rho.hi != 0;
    }
    static bool ConsiderPairExtrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      /*
      double abs_error = delta.d_rho.width() / 2;
      double rel_error_hi = abs_error / q_mass_result.rho.lo;
      
      if (rel_error_hi < AffinityCommon::Helpers::ErrorShare(
          param, q_mass_result.abs_error_used, r_node)) {
        q_postponed->abs_error_used += abs_error;
        q_postponed->d_rho += delta.d_rho.mid();
        return false;
      } else {
        return true;
      }
      */
      
      return true;
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
      return -delta.d_rho.lo;
    }
  };
};

void FindExemplars(index_t dimensionality, index_t n_points,
    CacheArray<AffinityAlpha::QPoint> *data_points) {
  ArrayList<Vector> exemplars;
  CacheReadIterator<AffinityAlpha::QPoint> point(data_points, 0);

  exemplars.Init();

  for (index_t point_i = 0; point_i < n_points; point_i++, point.Next()) {
    //ot::Print(point->info());
    if (point->info().rho > 0) {
      exemplars.AddBack()->Copy(point->vec());
    }
  }
  
  ot::Print(exemplars);

  Matrix m;
  m.Init(dimensionality, exemplars.size());

  for (index_t i = 0; i < exemplars.size(); i++) {
    Vector dest;
    m.MakeColumnVector(i, &dest);
    dest.CopyValues(exemplars[i]);
  }

  data::Save("exemplars.txt", m);
}

void FindCovariance(const Matrix& dataset) {
  Matrix m;
  Vector sums;

  m.Init(dataset.n_rows()-1, dataset.n_cols());
  sums.Init(dataset.n_rows() - 1);
  sums.SetZero();

  for (index_t i = 0; i < dataset.n_cols(); i++) {
    Vector s;
    Vector d;
    dataset.MakeColumnSubvector(i, 0, dataset.n_rows()-1, &s);
    m.MakeColumnVector(i, &d);
    d.CopyValues(s);
    la::AddTo(s, &sums);
  }
  
  la::Scale(-1.0 / dataset.n_cols(), &sums);
  for (index_t i = 0; i < dataset.n_cols(); i++) {
    Vector d;
    m.MakeColumnVector(i, &d);
    la::AddTo(sums, &d);
  }
  
  Matrix cov;

  la::MulTransBInit(m, m, &cov);
  la::Scale(1.0 / (dataset.n_cols() - 1), &cov);

  cov.PrintDebug("covariance");

  Vector d;
  Matrix u; // eigenvectors
  Matrix ui; // the inverse of eigenvectors

  //cov.PrintDebug("cov");
  la::EigenvectorsInit(cov, &d, &u);
  d.PrintDebug("covariance_eigenvectors");
  la::TransposeInit(u, &ui);

//
//  for (index_t i = 0; i < d.length(); i++) {
//    d[i] = 1.0 / sqrt(d[i]);
//  }
//
//  la::ScaleRows(d, &ui);
//
//  Matrix cov_inv_half;
//  la::MulInit(u, ui, &cov_inv_half);
//
//  Matrix final;
//  la::MulInit(cov_inv_half, m, &final);
//
//  for (index_t i = 0; i < dataset.n_cols(); i++) {
//    Vector s;
//    Vector d;
//    dataset.MakeColumnSubvector(i, 0, dataset.n_rows()-1, &d);
//    final.MakeColumnVector(i, &s);
//    d.CopyValues(s);
//  }
}

void AffinityMain(datanode *module, const char *gnp_name) {
  AffinityAlpha::Param param;

  param.Init(fx_submodule(module, gnp_name, gnp_name));

  TempCacheArray<AffinityAlpha::QPoint> data_points;
  TempCacheArray<AffinityAlpha::QNode> data_nodes;

  index_t n_block_points = fx_param_int(
      module, "n_block_points", 1024);
  index_t n_block_nodes = fx_param_int(
      module, "n_block_nodes", 128);

  datanode *data_module = fx_submodule(module, "data", "data");

  fx_timer_start(data_module, "read");

  Matrix data_matrix;
  MUST_PASS(data::Load(fx_param_str_req(data_module, ""), &data_matrix));
  index_t n_points = data_matrix.n_cols();
  index_t dimensionality = data_matrix.n_rows();
  AffinityAlpha::QPoint default_point;
  default_point.vec().Init(data_matrix.n_rows());
  param.BootstrapMonochromatic(&default_point, data_matrix.n_cols());
  data_points.Init(default_point, data_matrix.n_cols(), n_block_points);
  for (index_t i = 0; i < data_matrix.n_cols(); i++) {
    CacheWrite<AffinityAlpha::QPoint> point(&data_points, i);
    point->vec().CopyValues(data_matrix.GetColumnPtr(i));
  }
  FindCovariance(data_matrix);
  data_matrix.Destruct();
  data_matrix.Init(0, 0);

  fx_timer_stop(data_module, "read");

  AffinityAlpha::QNode data_example_node;
  data_example_node.Init(dimensionality, param);
  data_nodes.Init(data_example_node, 0, n_block_nodes);
  KdTreeMidpointBuilder
      <AffinityAlpha::QPoint, AffinityAlpha::QNode, AffinityAlpha::Param>
      ::Build(data_module, param, &data_points, &data_nodes);

  // All the above is not any different for affinity than for anything
  // else.  Now, time for the iteration.
  int n_iter = fx_param_int(module, "n_iter", 1000);
  int stable_iterations = 0;
  index_t n_changed = n_points / 2;

  ArrayList<char> is_exemplar;
  is_exemplar.Init(n_points);
  
  fprintf(stderr, " --- killing alpha ---\n");
  for (index_t i = 0; i < n_points; i++) {
    CacheWrite<AffinityAlpha::QPoint> point(&data_points, i);
    point->info().alpha.max1 = 0;
    point->info().alpha.max2 = param.pref;
    point->info().alpha.max1_index = i;
    point->info().rho = 0;
    is_exemplar[i] = 0;
  }

  for (int iter = 0; iter < n_iter && stable_iterations < 50; iter++)
  {
    double temperature = 0;//1.0 * n_changed / n_points / (n_iter + 1);
    double lambda = param.lambda * (1.0 - temperature) + 0.5 * temperature;
    double nonlambda = 1.0 - lambda;
    //double nonlambda2 = nonlambda * nonlambda;
    //double lambda2 = 1.0 - nonlambda2;
    index_t n_alpha_changed = 0;
    index_t n_exemplars = 0;

    {
      TempCacheArray<AffinityAlpha::QResult> q_results_alpha;

      AffinityAlpha::QResult default_result_alpha;
      default_result_alpha.Init(param);
      q_results_alpha.Init(default_result_alpha, data_points.end_index(),
          data_points.n_block_elems());

      nbr_utils::ThreadedDualTreeSolver
               < AffinityAlpha, DualTreeDepthFirst<AffinityAlpha> >::Solve(
          fx_submodule(module, "threads", "iter%d_alpha", iter), param,
          &data_points, &data_nodes, &data_points, &data_nodes,
          &q_results_alpha);

      for (index_t i = 0; i < n_points; i++) {
        CacheWrite<AffinityAlpha::QPoint> point(&data_points, i);
        CacheRead<AffinityAlpha::QResult> alpha(&q_results_alpha, i);

        if (point->info().alpha.max1_index != alpha->alpha.max1_index) {
          n_alpha_changed++;
        }

        point->info().alpha = alpha->alpha;

        //Re-implement damping
        //Look at the original algorithm
      }
      nbr_utils::StatFixer<
          AffinityAlpha::Param, AffinityAlpha::QPoint, AffinityAlpha::QNode>
          ::Fix(param, &data_points, &data_nodes);
    }

    {
      TempCacheArray<AffinityRho::QResult> q_results_rho;

      AffinityRho::QResult default_result_rho;
      default_result_rho.Init(param);
      q_results_rho.Init(default_result_rho, data_points.end_index(),
          data_points.n_block_elems());

      nbr_utils::ThreadedDualTreeSolver< AffinityRho, DualTreeDepthFirst<AffinityRho> >::Solve(
          fx_submodule(module, "threads", "iter%d_rho", iter), param,
          &data_points, &data_nodes, &data_points, &data_nodes,
          &q_results_rho);

      n_changed = 0;

      for (index_t i = 0; i < n_points; i++) {
        CacheWrite<AffinityAlpha::QPoint> point(&data_points, i);
        CacheRead<AffinityRho::QResult> compute_rho(&q_results_rho, i);
        double old_rho = point->info().rho;
        double new_rho = old_rho*lambda + compute_rho->rho*nonlambda;

        if (1) {
          point->info().rho = new_rho;
        } else {
          new_rho = point->info().rho;
        }

        if ((old_rho > 0) != (new_rho > 0)) {
          point->info().rho *= math::Random(0.99, 1.01);
           n_changed++;
        }

        if (new_rho > 0) {
          is_exemplar[i] = 1;
          n_exemplars++;
        } else {
          is_exemplar[i] = 0;
        }
      }

      if (n_changed == 0) {
        stable_iterations++;
      } else {
        stable_iterations = 0;
      }

      nbr_utils::StatFixer<
          AffinityAlpha::Param, AffinityAlpha::QPoint, AffinityAlpha::QNode>
          ::Fix(param, &data_points, &data_nodes);
    }

    fprintf(stderr, "------------- iter %04d: %"LI"d rhos changed, (%"LI"d exemplars), %d alphas\n",
        iter, n_changed, n_exemplars, n_alpha_changed);
  }

  FindExemplars(dimensionality, n_points, &data_points);
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  AffinityMain(fx_root, "affinity");

  fx_done();
}
