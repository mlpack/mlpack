#include "spbounds.h"
#include "gnp.h"
#include "dfs.h"
#include "nbr_utils.h"

#include "fastlib/fastlib.h"

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

#define SIMILARITY_MAX 0

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
    /** The epsilon for approximation for rho - ABSOLUTE error. */
    double eps;
    /** Epsilon per point. */
    double eps_per_point;
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
      OT_MY_OBJECT(eps_per_point);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(n_points);
      OT_MY_OBJECT(pref);
      OT_MY_OBJECT(lambda);
    }

   public:
    void Copy(const Param& other) {
      eps = other.eps;
      eps_per_point = other.eps_per_point;
      dim = other.dim;
      n_points = other.n_points;
      pref = other.pref;
      lambda = other.lambda;
    }

    void Init(datanode *module) {
      dim = -1;
      eps = fx_param_double(module, "eps", 0.0);
      pref = fx_param_double_req(module, "pref");
      lambda = fx_param_double(module, "lambda", 0.8);
    }

    void BootstrapMonochromatic(CombinedPoint *point, index_t count) {
      dim = point->vec().length();
      n_points = count;
      // NOTE: These values are manually assigned to be different later.
      point->info().rho = DBL_NAN;
      point->info().alpha.max1 = DBL_NAN;
      point->info().alpha.max2 = DBL_NAN;
      point->info().alpha.max1_index = -1;
      eps_per_point = eps / n_points;
    }
    
    void SetEpsilon(double eps_in) {
      eps = eps_in;
      eps_per_point = eps / n_points;
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
    static double SimilarityHi(
        const Param& param,
        const Vector& a, index_t a_index, const CombinedNode& b) {
      double distsq = b.bound().MinDistanceSqToPoint(a);
      double hi = Similarity(distsq);
      if (a_index < b.end() && a_index >= b.begin()
          && param.pref > hi) {
        hi = param.pref;
      }
      return hi;
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
      alpha.max1 = -DBL_MAX;
      alpha.max2 = -DBL_MAX;
      alpha.max1_index = -1;
    }
    void Postprocess(const Param& param,
        const QPoint& q, index_t q_index, const RNode& r_root) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QPoint& q, index_t q_i) {}
  };

  struct QMassResult {
   public:
    SpRange alpha;

    OT_DEF(QMassResult) {
      OT_MY_OBJECT(alpha);
    }

   public:
    void Init(const Param& param) {
      alpha.Init(-DBL_MAX, -DBL_MAX);
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

  struct PairVisitor {
   public:
    Alpha old_alpha;
    Alpha alpha;

   public:
    void Init(const Param& param) {}

    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_node, const QMassResult& unapplied_mass_results,
        QResult* q_result, GlobalResult* global_result) {
      double alpha_hi;

      alpha = q_result->alpha;
      old_alpha = q.info().alpha;

      if (unlikely(q_index >= r_node.begin() && q_index < r_node.end())) {
        alpha_hi = r_node.stat().rho.hi + old_alpha.max1;
      } else {
        double sim = AffinityCommon::Helpers::Similarity(
            r_node.bound().MinDistanceSqToPoint(q.vec()));
        alpha_hi = min(
            min(sim, old_alpha.max1) + r_node.stat().rho.hi,
            sim);
      }

      return (alpha_hi > alpha.max2);
    }
    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index, const RPoint& r, index_t r_index) {
      double candidate_alpha;

      if (likely(q_index != r_index)) {
        double sim = AffinityCommon::Helpers::Similarity(
            la::DistanceSqEuclidean(q.vec(), r.vec()));
        candidate_alpha = min(
            min(sim, old_alpha.get(r_index)) + r.info().rho,
            sim);
      } else {
        candidate_alpha = r.info().rho + old_alpha.get(r_index);
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
        const QPoint& q, index_t q_index,
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
      double sim_lo = AffinityCommon::Helpers::SimilarityLo(
          param, q_node, r_node);
      delta->alpha.lo = min(min(q_node.stat().alpha.lo, sim_lo)
          + r_node.stat().rho.lo, sim_lo);

      if (q_node.begin() < r_node.end() && r_node.begin() < q_node.end()) {
        delta->alpha.hi =
            q_node.stat().alpha.hi + r_node.stat().rho.hi;
      } else {
        double sim_hi = AffinityCommon::Helpers::Similarity(
            q_node.bound().MinDistanceSqToBound(r_node.bound()));
        delta->alpha.hi = min(min(q_node.stat().alpha.hi, sim_hi)
            + r_node.stat().rho.hi, sim_hi);
      }

      return true;
    }
    static bool ConsiderPairExtrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return (delta.alpha.hi > q_mass_result.alpha.lo);
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

#ifdef APPROX
  struct QPostponed {
   public:
    double d_rho;
    double error_buffer;

    OT_DEF(QPostponed) {
      OT_MY_OBJECT(d_rho);
      OT_MY_OBJECT(error_buffer);
    }

   public:
    void Init(const Param& param) {
      Reset(param);
    }

    void Reset(const Param& param) {
      d_rho = 0;
      error_buffer = 0;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      d_rho += other.d_rho;
      error_buffer += other.error_buffer;
    }
  };
#else
  typedef BlankQPostponed QPostponed;
#endif

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
    #ifdef APPROX
    double error_buffer;
    #endif

    OT_DEF(QResult) {
      OT_MY_OBJECT(rho);
      #ifdef APPROX
      OT_MY_OBJECT(error_buffer);
      #endif
    }

   public:
    void Init(const Param& param) {
      rho = 0;
      #ifdef APPROX
      error_buffer = 0;
      #endif
    }
    void Postprocess(const Param& param,
        const QPoint& q, index_t q_index, const RNode& r_root) {
      double self_responsibility = param.pref - q.info().alpha.get(q_index);
      rho += self_responsibility;
    }
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QPoint& q, index_t q_index) {
      #ifdef APPRLX
      rho += postponed.d_rho;
      error_buffer += postponed.error_buffer;
      #endif
    }
  };

#ifdef APPROX
  struct QMassResult {
   public:
    SpRange rho;
    double error_buffer;

    OT_DEF(QMassResult) {
      OT_MY_OBJECT(rho);
      OT_MY_OBJECT(error_buffer);
    }

   public:
    void Init(const Param& param) {
      rho.Init(0, 0);
      error_buffer = 0;
    }
    void ApplyMassResult(const Param& param, const QMassResult& mass_result) {
      rho += mass_result.rho;
      error_buffer += mass_result.error_buffer;
    }
    void ApplyDelta(const Param& param, const Delta& delta) {
      rho += delta.d_rho;
    }
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      rho += postponed.d_rho;
      error_buffer += postponed.error_buffer;
    }
    void StartReaccumulate(const Param& param, const QNode& q_node) {
      rho.InitEmptySet();
      error_buffer = DBL_MAX;
    }
    void Accumulate(const Param& param, const QResult& result) {
      rho |= result.rho;
      error_buffer = min(error_buffer, result.error_buffer);
    }
    void Accumulate(const Param& param,
        const QMassResult& result, index_t n_points) {
      rho |= result.rho;
      error_buffer = min(error_buffer, result.error_buffer);
    }
    void FinishReaccumulate(const Param& param, const QNode& q_node) {}
  };
#else
  typedef BlankQMassResult QMassResult;
#endif

  struct PairVisitor {
   public:
    double rho;

   public:
    void Init(const Param& param) {}
    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_node, const QMassResult& unapplied_mass_results,
        QResult* q_result, GlobalResult* global_result) {
      // do the point-node prune check
      double sim_hi = AffinityCommon::Helpers::SimilarityHi(
          param, q.vec(), q_index, r_node);
      #ifdef APPROX
      q_result->error_buffer += param.eps_per_point * r_node.count();
      #endif
      if (sim_hi < r_node.stat().alpha.lo) {
        return false;
      } else {
        rho = q_result->rho;
        return true;
      }
    }
    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
      if (likely(q_index != r_index)) {
        double responsibility =
            AffinityCommon::Helpers::Similarity(q.vec(), r.vec())
            - r.info().alpha.get(q_index);
        if (responsibility > 0) {
          rho += responsibility;
        }
      }
    }
    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
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
      #ifdef APPROX
      double my_abs_error = delta.d_rho.width() * 0.5;
      double allotted_error = param.eps_per_point * r_node.count();
      double allowed_error = allotted_error + q_mass_result.error_buffer;
      
      //if (param.eps != 0)
      //fprintf(stderr, "%g %g %g %g\n", my_abs_error, param.eps_per_point, allotted_error, q_mass_result.error_buffer);
      
      if (my_abs_error <= allowed_error) {
        q_postponed->error_buffer += allotted_error - my_abs_error;
        q_postponed->d_rho += delta.d_rho.mid();
        return false;
      } else {
        return true;
      }
      #else
      return true;
      #endif
    }
    static bool ConsiderQueryTermination(const Param& param,
        const QNode& q_node,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
    }
    static double Heuristic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta) {
      // TODO: If approximating, favor upper bound
      return -delta.d_rho.hi;
    }
  };
};

struct Cluster {
  Vector exemplar;
  Vector centroid;
  index_t count;
};

void FindExemplars(datanode *module, const AffinityCommon::Param& param,
    index_t dimensionality, index_t n_points,
    CacheArray<AffinityAlpha::QPoint> *data_points) {
  fx_timer_start(module, "exemplars");

  ArrayList<Cluster> clusters;
  ArrayList<index_t> assignments;
  CacheReadIterator<AffinityAlpha::QPoint> point(data_points, 0);

  clusters.Init();

  for (index_t point_i = 0; point_i < n_points; point_i++, point.Next()) {
    if (point->info().rho > 0) {
      Cluster *cluster = clusters.AddBack();
      cluster->exemplar.Copy(point->vec());
      cluster->centroid.Init(dimensionality);
      cluster->centroid.SetZero();
      cluster->count = 0;
    }
  }

  // Run all nearest neighbors to assign clusters and find centroids
  assignments.Init(n_points);

  point.SetIndex(0);
  for (index_t point_i = 0; point_i < n_points; point_i++, point.Next()) {
    double best_distsq = DBL_MAX;
    index_t best_k = -1;

    for (index_t k = 0; k < clusters.size(); k++) {
      double distsq = la::DistanceSqEuclidean(
          clusters[k].exemplar, point->vec());
      if (unlikely(distsq < best_distsq)) {
        best_distsq = distsq;
        best_k = k;
      }
    }

    assignments[point_i] = best_k;
    clusters[best_k].count++;
    la::AddTo(point->vec(), &clusters[best_k].centroid);
  }

  // Divide centroids by size
  for (index_t i = 0; i < clusters.size(); i++) {
    la::Scale(1.0 / clusters[i].count, &clusters[i].centroid);
  }

  // Calculate net similarity
  double netsim = param.pref * clusters.size();

  point.SetIndex(0);
  for (index_t point_i = 0; point_i < n_points; point_i++, point.Next()) {
    Cluster *cluster = &clusters[assignments[point_i]];
    double distsq = la::DistanceSqEuclidean(cluster->exemplar, point->vec());
    netsim -= distsq;
  }

  fx_timer_start(module, "exemplars");
  fx_format_result(module, "netsim", "%f", netsim);

  // Make a matrix of exemplars so we can save it to file
  Matrix m;
  m.Init(dimensionality, clusters.size());

  for (index_t i = 0; i < clusters.size(); i++) {
    Vector dest;
    m.MakeColumnVector(i, &dest);
    dest.CopyValues(clusters[i].exemplar);
  }

  data::Save("exemplars.txt", m);
}

void FindCovariance(const Matrix& dataset) {
  Matrix m;
  Vector sums;

  m.Init(dataset.n_rows(), dataset.n_cols());
  sums.Init(dataset.n_rows());
  sums.SetZero();

  for (index_t i = 0; i < dataset.n_cols(); i++) {
    Vector s;
    Vector d;
    dataset.MakeColumnSubvector(i, 0, dataset.n_rows(), &s);
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

  la::EigenvectorsInit(cov, &d, &u);
  d.PrintDebug("covariance_eigenvectors");
  la::TransposeInit(u, &ui);

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

inline double damp(double lambda, double prev, double next) {
  return lambda * prev + (1 - lambda) * next;
}

void TimeStats(datanode *module, const ArrayList<double>& list) {
  double v_avg;
  MinHeap<double, char> heap;

  heap.Init();
  v_avg = 0;
  for (index_t i = 0; i < list.size(); i++) {
    v_avg += list[i];
    heap.Put(list[i], 0);
  }
  v_avg /= list.size();

  double v_min;
  double v_med;
  double v_max;

  v_min = heap.top_key();
  while (heap.size() > list.size() / 2) {
    heap.PopOnly();
  }
  v_med = heap.top_key();
  while (heap.size() > 1) {
    heap.PopOnly();
  }
  v_max = heap.top_key();

  fx_format_result(module, "min", "%f", v_min/1e6);
  fx_format_result(module, "med", "%f", v_med/1e6);
  fx_format_result(module, "max", "%f", v_max/1e6);
  fx_format_result(module, "avg", "%f", v_avg/1e6);
  fx_format_result(module, "sum", "%f", v_avg*list.size()/1e6);
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
  data_points.Init(default_point, n_points, n_block_points);
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

  // Initial conditions for alpha
  for (index_t i = 0; i < n_points; i++) {
    CacheWrite<AffinityAlpha::QPoint> point(&data_points, i);
    point->info().alpha.max1 = 0;
    point->info().alpha.max2 = param.pref;
    point->info().alpha.max1_index = i;
    if (rand() % 2 == 0) {
      point->info().rho = -param.pref / 2;
    } else {
      point->info().rho = 0;
    }
  }
  data_nodes.Init(data_example_node, 0, n_block_nodes);
  KdTreeMidpointBuilder
      <AffinityAlpha::QPoint, AffinityAlpha::QNode, AffinityAlpha::Param>
      ::Build(data_module, param, &data_points, &data_nodes);


  // ---------------------------------------------------------------
  // The algorithm follows -----------------------------------------

  // All the above is not any different for affinity than for anything
  // else.  Now, time for the iteration.
  int n_iter = fx_param_int(module, "n_iter", 10000);
  int iter;
  int n_convergence_iter = fx_param_int(module, "n_convergence_iter", 16);
  int convergence_thresh = fx_param_int(module, "convergence_thresh", 8);
  int stable_iter = 0;
  index_t n_changed = n_points / 2;
  index_t n_exemplars = 0;
  timer *timer_rho = fx_timer(module, "all_rho");
  timer *timer_alpha = fx_timer(module, "all_alpha");

  ArrayList<char> is_exemplar;
  is_exemplar.Init(n_points);
  
  ArrayList<double> iter_times_rho;
  ArrayList<double> iter_times_alpha;
  ArrayList<double> iter_times_total;
  iter_times_rho.Init();
  iter_times_alpha.Init();
  iter_times_total.Init();

  for (index_t i = 0; i < n_points; i++) {
    is_exemplar[i] = 0;
  }

  for (iter = 0; iter < n_iter; iter++)
  {
    double lambda = param.lambda;
    index_t n_alpha_changed = 0;
    double sum_alpha = 0;
    double sum_alpha2 = 0;
    double sum_rho = 0;
    index_t unclassifieds;
    double last_rho_time = timer_rho->total.micros;
    double last_alpha_time = timer_alpha->total.micros;

    n_exemplars = 0;
    unclassifieds = 0;

    fx_timer_start(module, "all_alpha");
    {
      TempCacheArray<AffinityAlpha::QResult> q_results_alpha;

      AffinityAlpha::QResult default_result_alpha;
      default_result_alpha.Init(param);
      q_results_alpha.Init(default_result_alpha, data_points.end_index(),
          data_points.n_block_elems());

      nbr_utils::ThreadedDualTreeSolver
               < AffinityAlpha, DualTreeDepthFirst<AffinityAlpha> >::Solve(
          fx_submodule(module, "threads", "iters/%d/alpha", iter), param,
          &data_points, &data_nodes, &data_points, &data_nodes,
          &q_results_alpha);

      for (index_t i = 0; i < n_points; i++) {
        CacheWrite<AffinityAlpha::QPoint> point(&data_points, i);
        CacheRead<AffinityAlpha::QResult> result(&q_results_alpha, i);

        if (point->info().alpha.max1_index != result->alpha.max1_index) {
          n_alpha_changed++;
        }

        point->info().alpha = result->alpha;

        sum_alpha += result->alpha.max1;
        sum_alpha2 += result->alpha.max2;

        if (!is_exemplar[result->alpha.max1_index] && point->info().rho <= 0) {
          unclassifieds++;
        }
      }

      nbr_utils::StatFixer<
          AffinityAlpha::Param, AffinityAlpha::QPoint, AffinityAlpha::QNode>
          ::Fix(param, &data_points, &data_nodes);
    }
    fx_timer_stop(module, "all_alpha");

    fprintf(stderr,
        "\033[31m     -------- %04d alpha: sum_alpha = %f, sum_alpha2 = %f, %"LI"d alphas changed, %"LI"d unclassifieds, %.3fsec cum. alpha\033[0m\n",
        iter, sum_alpha, sum_alpha2, n_alpha_changed, unclassifieds,
        timer_alpha->total.micros / 1.0e6);

    fx_timer_start(module, "all_rho");
    {
      double rho_squared_difference = 0;
      TempCacheArray<AffinityRho::QResult> q_results_rho;

      AffinityRho::QResult default_result_rho;
      default_result_rho.Init(param);
      q_results_rho.Init(default_result_rho, data_points.end_index(),
          data_points.n_block_elems());

      nbr_utils::ThreadedDualTreeSolver< AffinityRho, DualTreeDepthFirst<AffinityRho> >::Solve(
          fx_submodule(module, "threads", "iters/%d/rho", iter), param,
          &data_points, &data_nodes, &data_points, &data_nodes,
          &q_results_rho);

      n_changed = 0;

      for (index_t i = 0; i < n_points; i++) {
        CacheWrite<AffinityAlpha::QPoint> point(&data_points, i);
        CacheRead<AffinityRho::QResult> result(&q_results_rho, i);
        double old_rho = point->info().rho;
        double new_rho = damp(lambda, old_rho, result->rho);

        if ((old_rho > 0) != (new_rho > 0)) {
          new_rho *= math::Random(0.4, 1.6);
          n_changed++;
        }

        rho_squared_difference += math::Sqr(new_rho - old_rho);
        sum_rho += new_rho;

        if (new_rho > 0) {
          is_exemplar[i] = 1;
          n_exemplars++;
        } else {
          is_exemplar[i] = 0;
        }

        point->info().rho = new_rho;
      }


      nbr_utils::StatFixer<
          AffinityAlpha::Param, AffinityAlpha::QPoint, AffinityAlpha::QNode>
          ::Fix(param, &data_points, &data_nodes);

      param.SetEpsilon(sqrt(rho_squared_difference / n_points));
      //param.SetEpsilon(0);
    }
    fx_timer_stop(module, "all_rho");

    fprintf(stderr, "\033[32m------------- %04d rho: %"LI"d rhos changed, (%"LI"d exemplars), sum_rho = %f, eps = %f, %.3fsec cum. rho\033[0m\n",
        iter, n_changed, n_exemplars,
        sum_rho, param.eps,
        timer_rho->total.micros / 1.0e6);

    *iter_times_rho.AddBack() = timer_rho->total.micros - last_rho_time;
    *iter_times_alpha.AddBack() = timer_alpha->total.micros - last_alpha_time;
    *iter_times_total.AddBack() = (timer_rho->total.micros - last_rho_time)
        + (timer_alpha->total.micros - last_alpha_time);

    if (n_changed < convergence_thresh) {
      stable_iter++;
    } else {
      stable_iter = 0;
    }
    
    if (stable_iter >= n_convergence_iter && iter > 20) {
      break;
    }
  }

  fx_format_result(module, "n_iterations", "%d", iter);
  fx_format_result(module, "n_exemplars", "%d", n_exemplars);

  TimeStats(fx_submodule(module, "iter_times_rho", "iter_times_rho"),
      iter_times_rho);
  TimeStats(fx_submodule(module, "iter_times_alpha", "iter_times_alpha"),
      iter_times_alpha);
  TimeStats(fx_submodule(module, "iter_times_total", "iter_times_total"),
      iter_times_total);
  fx_format_result(module, "n_points", "%"LI"d", n_points);

  // This will take too long if there are too many exemplars.
  if (n_exemplars >= 10000) {
    NONFATAL("Too many exemplars (%"LI"d >= 10000), NOT performing clustering.\n",
        n_exemplars);
  } else {
    FindExemplars(module, param, dimensionality, n_points, &data_points);
  }
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand(time(NULL));

  AffinityMain(fx_root, "affinity");

  fx_done();
}
