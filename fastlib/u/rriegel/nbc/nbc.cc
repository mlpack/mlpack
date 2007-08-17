#include "fastlib/fastlib_int.h"
#include "thor/thor.h"

/**
 * Nonparametric Bayes Classification for 2 classes.
 *
 * Uses the Epanechnikov kernel and provided priors.
 */
class Nbc {
 public:
  /** The bounding type. Required by THOR. */
  typedef DHrectBound<2> Bound;

  /** Point data includes class (referencs) and prior (queries). */
  class NbcPoint {
   private:
    Vector vec_;
    /** The point's class (if reference). */
    bool is_pos_;
    /** The point's prior (if query). */
    double pi_;

    OT_DEF_BASIC(NbcPoint) {
      OT_MY_OBJECT(vec_);
    }

   public:

    /**
     * Gets the vector.
     */
    const Vector& vec() const {
      return vec_;
    }
    /**
     * Gets the vector.
     */
    Vector& vec() {
      return vec_;
    }
    /**
     * Gets the class.
     */
    bool is_pos() const {
      return is_pos_;
    }
    /**
     * Gets the positive prior.
     */
    double pi_pos() const {
      return pi_;
    }
    /**
     * Gets the negative prior.
     */
    double pi_neg() const {
      return 1 - pi_;
    }
    /**
     * Initializes a "default element" from a dataset schema.
     *
     * This is the only function that allows allocation.
     */
    template<typename Param>
    void Init(const Param& param, const DatasetInfo& schema) {
      vec_.Init(schema.n_features() - 2); // 
      vec_.SetZero();
      is_pos_ = false;
      pi_ = -1.0;
    }
    /**
     * Sets the values of this object, not allocating any memory.
     *
     * If memory needs to be allocated it must be allocated at the beginning
     * with Init.
     *
     * @param param ignored
     * @param data the vector data read from file
     */
    template<typename Param>
    void Set(const Param& param, const Vector& data) {
      mem::Copy(vec_.ptr(), data.ptr(), vec_.length());
      is_pos_ = (data[data.length() - 2] != 0.0);
      pi_ = data[data.length() - 1];
    }
  };
  typedef NbcPoint QPoint;
  typedef NbcPoint RPoint;

  /** The type of kernel in use.  NOT required by THOR. */
  typedef EpanKernel Kernel;

  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by THOR.
   */
  struct Param {
   public:
    /**
     * Normalized threshold for positive points. Similar for _neg.
     *
     * Pre-normalization, thresh_pos.lo = 1 - thresh_neg.hi, etc.
     */
    DRange norm_thresh_pos;
    DRange norm_thresh_neg;
    /** The kernel for positive points. Similar for _neg. */
    Kernel kernel_pos;
    Kernel kernel_neg;
    /** The dimensionality of the data sets. */
    index_t dim;
    /** Number of reference points. */
    index_t count_all;
    /** Number of positive reference points. Similar for _neg. */
    index_t count_pos;
    index_t count_neg;
    /** The original threshold, before normalization. */
    double orig_thres;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(norm_thresh_pos);
      OT_MY_OBJECT(norm_thresh_neg);
      OT_MY_OBJECT(kernel_pos);
      OT_MY_OBJECT(kernel_neg);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(count_all);
      OT_MY_OBJECT(count_pos);
      OT_MY_OBJECT(count_neg);
      OT_MY_OBJECT(orig_thresh);
    }

   public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      kernel_pos.Init(fx_param_double_req(module, "hpos"));
      kernel_neg.Init(fx_param_double_req(module, "hneg"));
      orig_thresh = fx_param_double(module, "threshold", "0.5");
    }

    void SetDimensions(index_t vector_dimension, index_t n_points) {
      dim = vector_dimension - 2; // last two cols: class, prior
      count_all = n_points;
      // TODO: this must be done after tree building
      //double normalized_threshold =
      //    nominal_threshold * kernel.CalcNormConstant(dim) * count;
      //thresh.lo = normalized_threshold * (1.0 - 1.0e-3);
      //thresh.hi = normalized_threshold * (1.0 + 1.0e-3);
      //ot::Print(dim);
      //ot::Print(kernel);
      //ot::Print(thresh);
    }

   public:
    // Convenience methods for purpose of thresholded KDE

    /**
     * Compute kernel sum for a region of reference points assuming we have
     * the actual query point (not THOR).
     */
    double ComputeKernelSum(
        const Vector& q,
        index_t r_count, const Vector& r_mass, double r_sumsq) const {
      double quadratic_term =
          + r_count * la::Dot(q, q)
          - 2.0 * la::Dot(q, r_mass)
          + r_sumsq;
      return r_count - quadratic_term * kernel.inv_bandwidth_sq();
    }

    /**
     * Divides a vector by an integer (not THOR).
     */
    static void ComputeCenter(
        index_t count, const Vector& mass, Vector* center) {
      DEBUG_ASSERT(count != 0);
      center->Copy(mass);
      la::Scale(1.0 / count, center);
    }

    /**
     * Compute kernel sum given only a squared distance (not THOR).
     */
    double ComputeKernelSum(
        double distance_squared,
        index_t r_count, const Vector& r_center, double r_sumsq) const {
      //q*q - 2qr + rsumsq
      //q*q - 2qr + r*r - r*r
      double quadratic_term =
          (distance_squared - la::Dot(r_center, r_center)) * r_count
          + r_sumsq;

      return -quadratic_term * kernel.inv_bandwidth_sq() + r_count;
    }
  };


  /**
   * Moment information used by thresholded KDE.
   *
   * NOT required by THOR, but used within other classes.
   */
  struct MomentInfo {
   public:
    Vector mass;
    double sumsq;
    index_t count;

    OT_DEF_BASIC(MomentInfo) {
      OT_MY_OBJECT(mass);
      OT_MY_OBJECT(sumsq);
      OT_MY_OBJECT(count);
    }

   public:
    void Init(const Param& param) {
      DEBUG_ASSERT(param.dim > 0);
      mass.Init(param.dim);
      Reset();
    }

    void Reset() {
      mass.SetZero();
      sumsq = 0;
      count = 0;
    }

    void Add(index_t count_in, const Vector& mass_in, double sumsq_in) {
      if (unlikely(count_in != 0)) {
        la::AddTo(mass_in, &mass);
        sumsq += sumsq_in;
        count += count_in;
      }
    }

    void Add(const MomentInfo& other) {
      Add(other.count, other.mass, other.sumsq);
    }

    double ComputeKernelSum(const Param& param, const Vector& point) const {
      return param.ComputeKernelSum(point, count, mass, sumsq);
    }

    DRange ComputeKernelSumRange(const Param& param,
        const Bound& query_bound) const {
      DRange density_bound;
      Vector center;

      param.ComputeCenter(count, mass, &center);

      density_bound.lo = param.ComputeKernelSum(
          query_bound.MaxDistanceSq(center),
          count, center, sumsq);
      density_bound.hi = param.ComputeKernelSum(
          query_bound.MinDistanceSq(center),
          count, center, sumsq);

      return density_bound;
    }

    bool is_empty() const {
      return likely(count == 0);
    }
  };

  /**
   * Per-node bottom-up statistic for both queries and references.
   *
   * The statistic must be commutative and associative, thus bottom-up
   * computable.
   *
   * Note that queries need only the pi bounds and references need
   * everything else, suggesting these could be in separate classes,
   * but QStat must equal RStat in order for us to allow monochromatic
   * execution.
   *
   * This limitation may be removed in a further version of THOR.
   */
  struct NbcStat {
   public:
    /** Data used in inclusion pruning. Similar for _neg. */
    MomentInfo moment_info_pos;
    MomentInfo moment_info_neg;
    /** Bounding box of the positive points. Similar for _neg. */
    Bound bound_pos;
    Bound bound_neg;
    /** Number of positive ponits. Similar for _neg. */
    index_t count_pos;
    index_t count_neg;
    /** Bounds for query priors. Similar for _neg. */
    DRange pi_pos;
    DRange pi_neg;

    OT_DEF_BASIC(NbcStat) {
      OT_MY_OBJECT(moment_info_pos);
      OT_MY_OBJECT(moment_info_neg);
      OT_MY_OBJECT(bound_pos);
      OT_MY_OBJECT(bound_neg);
      OT_MY_OBJECT(count_pos);
      OT_MY_OBJECT(count_neg);
    }

   public:
    /**
     * Initialize to a default zero value, as if no data is seen (Req THOR).
     *
     * This is the only method in which memory allocation can occur.
     */
    void Init(const Param& param) {
      moment_info_pos.Init(param);
      moment_info_neg.Init(param);
      bound_pos.Init(param.dim);
      bound_neg.Init(param.dim);
      count_pos = 0;
      count_neg = 0;
      pi_pos.InitEmptySet();
      pi_neg.InitEmptySet();
    }

    /**
     * Accumulate data from a single point (Req THOR).
     */
    void Accumulate(const Param& param, const NbcPoint& point) {
      if (point.is_pos()) {
	moment_info_pos.Add(1, point.vec(), la::Dot(point.vec(), point.vec()));
	bound_pos |= point.vec();
	++count_pos;
      } else {
	moment_info_neg.Add(1, point.vec(), la::Dot(point.vec(), point.vec()));
	bound_neg |= point.vec();
	++count_neg;
      }
      pi_pos |= point.pi();
      pi_neg |= 1 - point.pi();
    }

    /**
     * Accumulate data from one of your children (Req THOR).
     */
    void Accumulate(const Param& param,
        const NbcStat& stat, const Bound& bound, index_t n) {
      moment_info_pos.Add(stat.moment_info_pos);
      moment_info_neg.Add(stat.moment_info_neg);
      bound_pos |= stat.bound_pos;
      bound_neg |= stat.bound_neg;
      count_pos += stat.count_pos;
      count_neg += stat.count_neg;
      pi_pos |= stat.pi_pos();
      pi_neg |= stat.pi_neg();
    }

    /**
     * Finish accumulating data; for instance, for mean, divide by the
     * number of points.
     */
    void Postprocess(const Param& param, const Bound& bound, index_t n) {
    }
  };
  typedef RStat NbcStat;
  typedef QStat NbcStat;
 
  /**
   * Reference node.
   */
  typedef ThorNode<Bound, RStat> RNode;
  /**
   * Query node.
   */
  typedef ThorNode<Bound, QStat> QNode;

  enum Label {
    LAB_NEITHER = 0,
    LAB_POS = 1,
    LAB_NEG = 2,
    LAB_EITHER = 3
  };

  /**
   * Coarse result on a region.
   */
  struct QPostponed {
   public:
    /** Moments of pruned things. */
    MomentInfo moment_info;
    /** We pruned an entire part of the tree with a particular label. */
    int label;

    OT_DEF_BASIC(QPostponed) {
      OT_MY_OBJECT(moment_info);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      moment_info.Init(param);
      label = LAB_EITHER;
    }

    void Reset(const Param& param) {
      moment_info.Reset();
      label = LAB_EITHER;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      label &= other.label;
      DEBUG_ASSERT_MSG(label != LAB_NEITHER, "Conflicting labels?");
      moment_info.Add(other.moment_info);
    }
  };

  /**
   * Coarse result on a region.
   */
  struct Delta {
   public:
    /** Density update to apply to children's bound. Similar for _neg. */
    DRange d_density_pos;
    DRange d_density_neg;

    OT_DEF_BASIC(Delta) {
      OT_MY_OBJECT(d_density_pos);
      OT_MY_OBJECT(d_density_neg);
    }

   public:
    void Init(const Param& param) {
    }
  };

  // rho
  struct QResult {
   public:
    double density_pos;
    double density_neg;
    int label;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(density_pos);
      OT_MY_OBJECT(density_neg);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      density_pos = 0.0;
      density_neg = 0.0;
      label = LAB_EITHER;
    }

    void Postprocess(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_root) {
      if (param.const_pos.lo * density_pos * q.pi_pos()
	  > param.const_neg.hi * density_neg * q.pi_neg()) {
        label &= LAB_POS;
      } else if (param.const_neg.lo * density_neg * q.pi_neg()
	  > param.const_pos.hi * density_pos * q.pi_pos()) {
        label &= LAB_NEG;
      }
      DEBUG_ASSERT_MSG(label != LAB_NEITHER,
          "Conflicting labels: [%g, %g]; %g > %g; %g > %g",
	  density_pos, density_neg,
	  param.const_pos.lo * density_pos * q.pi_pos(),
	  param.const_neg.hi * density_neg * q.pi_neg(),
	  param.const_neg.lo * density_neg * q.pi_neg(),
	  param.const_pos.hi * density_pos * q.pi_pos());
    }

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const QPoint& q, index_t q_index) {
      label &= postponed.label;
      DEBUG_ASSERT(label != LAB_NEITHER);

      if (!postponed.moment_info_pos.is_empty()) {
        density_pos += postponed.moment_info_pos.ComputeKernelSum(
            param, q.vec());
      }
      if (!postponed.moment_info_neg.is_empty()) {
        density_neg += postponed.moment_info_neg.ComputeKernelSum(
            param, q.vec());
      }
    }
  };

  struct QSummaryResult {
   public:
    /** Bound on density from leaves. Similar for _neg. */
    DRange density_pos;
    DRange density_neg;
    int label;

    OT_DEF_BASIC(QSummaryResult) {
      OT_MY_OBJECT(density_pos);
      OT_MY_OBJECT(density_neg);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      /* horizontal init */
      density_pos.Init(0, 0);
      density_neg.Init(0, 0);
      label = LAB_EITHER;
    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
      /* vertical init */
      density_pos.InitEmptySet();
      density_neg.InitEmptySet();
      label = LAB_NEITHER;
    }

    void Accumulate(const Param& param, const QResult& result) {
      // TODO: applying to single result could be made part of QResult,
      // but in some cases may require a copy/undo stage
      density_pos |= result.density_pos;
      density_neg |= result.density_neg;
      label |= result.label;
      DEBUG_ASSERT(result.label != LAB_NEITHER);
    }

    void Accumulate(const Param& param,
        const QSummaryResult& result, index_t n_points) {
      density_pos |= result.density_pos;
      density_neg |= result.density_neg;
      label |= result.label;
      DEBUG_ASSERT(result.label != LAB_NEITHER);
    }

    void FinishReaccumulate(const Param& param,
        const QNode& q_node) {
      /* no post-processing steps necessary */
    }

    /** horizontal join operator */
    void ApplySummaryResult(const Param& param,
        const QSummaryResult& summary_result) {
      density_pos += summary_result.density_pos;
      density_neg += summary_result.density_neg;
      label &= summary_result.label;
      DEBUG_ASSERT(label != LAB_NEITHER);
    }

    void ApplyDelta(const Param& param,
        const Delta& delta) {
      density_pos += delta.d_density_pos;
      density_neg += delta.d_density_neg;
    }

    bool ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      bool change_made;

      if (unlikely(postponed.label != LAB_EITHER)) {
        label &= postponed.label;
        DEBUG_ASSERT(label != LAB_NEITHER);
        change_made = true;
      }
      if (unlikely(!postponed.moment_info_pos.is_empty())) {
        density_pos += postponed.moment_info_pos.ComputeKernelSumRange(
            param, q_node.bound());
        change_made = true;
      }
      if (unlikely(!postponed.moment_info_neg.is_empty())) {
        density_neg += postponed.moment_info_neg.ComputeKernelSumRange(
            param, q_node.bound());
        change_made = true;
      }

      return change_made;
    }
  };

  /**
   * A simple postprocess-step global result.
   */
  struct GlobalResult {
   public:
    index_t count_pos;
    index_t count_unknown;
    
    OT_DEF(GlobalResult) {
      OT_MY_OBJECT(count_pos);
      OT_MY_OBJECT(count_unknown);
    }

   public:
    void Init(const Param& param) {
      count_pos = 0;
      count_unknown = 0;
    }
    void Accumulate(const Param& param, const GlobalResult& other) {
      count_pos += other.count_pos;
      count_unknown += other.count_unknown;
    }
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}
    void Postprocess(const Param& param) {}
    void Report(const Param& param, datanode *datanode) {
      fx_format_result(datanode, "count_pos", "%"LI"d",
          count_pos);
      fx_format_result(datanode, "percent_pos", "%.05f",
          double(count_pos) / param.count * 100.0);
      fx_format_result(datanode, "count_unknown", "%"LI"d",
          count_unknown);
      fx_format_result(datanode, "percent_unknown", "%.05f",
          double(count_unknown) / param.count * 100.0);
    }
    void ApplyResult(const Param& param,
        const QPoint& q_point, index_t q_i,
        const QResult& result) {
      fflush(stderr);
      if (result.label == LAB_POS) {
        ++count_pos;
      } else if (result.label == LAB_EITHER) {
        ++count_unknown;
      }
    }
  };

  /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  struct PairVisitor {
   public:
    double density_pos;
    double density_neg;
    bool do_pos;
    bool do_neg;

   public:
    void Init(const Param& param) {}

    // notes
    // - this function must assume that global_result is incomplete (which is
    // reasonable in allnn)
    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
      if (unlikely(q_result->label != LAB_EITHER)) {
        return false;
      }

      if (unlikely(
	   (r_node.stat().count_pos == 0 ||
	    r_node.stat().bound_pos.MinDistanceSq(q.vec())
	    > param.kernel_pos.bandwidth_sq()) &&
	   (r_node.stat().count_neg == 0 ||
	    r_node.stat().bound_neg.MinDistanceSq(q.vec())
	    > param.kernel_neg.bandwidth_sq()))) {
	return false;
      }

      density_pos = 0.0;
      density_neg = 0.0;
      do_pos = true;
      do_neg = true;
      if (r_node.stat().count_pos > 0) {
	if (unlikely(
	     r_node.stat().bound_pos.MaxDistanceSq(q.vec())
	     < param.kernel_pos.bandwidth_sq())) {
	  q_result->density_pos +=
	    r_node.stat().moment_info_pos.ComputeKernelSum(
		param, q.vec());
	  do_pos = false;
	}
      }
      if (r_node.stat().count_neg > 0) {
	if (unlikely(
	     r_node.stat().bound_neg.MaxDistanceSq(q.vec())
	     < param.kernel_neg.bandwidth_sq())) {
	  q_result->density_neg +=
	    r_node.stat().moment_info_neg.ComputeKernelSum(
		param, q.vec());
	  do_neg = false;
	}
      }

      return do_pos || do_neg;
    }

    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
      if (r.is_pos()) {
	if (likely(do_pos)) {
	  double distance = la::DistanceSqEuclidean(q.vec(), r.vec());
	  density_pos += param.kernel_pos.EvalUnnormOnSq(distance);
	}
      } else {
	if (likely(do_neg)) {
	  double distance = la::DistanceSqEuclidean(q.vec(), r.vec());
	  density_neg += param.kernel_neg.EvalUnnormOnSq(distance);
	}
      }
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
      q_result->density_pos += density_pos;
      q_result->density_neg += density_neg;
      
      DRange total_density_pos =
          unapplied_summary_results.density_pos + q_result->density_pos;
      DRange total_density_neg =
          unapplied_summary_results.density_neg + q_result->density_neg;

      if (unlikely(
	   param.const_pos.lo * total_density_pos.lo * q.pi_pos()
	   > param.const_neg.hi * total_density_neg.hi * q.pi_neg())) {
        label &= LAB_POS;
      } else if (unlikely(
	   param.const_neg.lo * total_density_neg.lo * q.pi_neg()
	   > param.const_pos.hi * total_density_pos.hi * q.pi_pos())) {
        label &= LAB_NEG;
      }
    }
  };

  class Algorithm {
   public:
    /**
     * Calculates a delta....
     *
     * - If this returns true, delta is calculated, and global_result is
     * updated.  q_postponed is not touched.
     * - If this returns false, delta is not touched.
     */
    static bool ConsiderPairIntrinsic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        Delta* delta,
        GlobalResult* global_result,
        QPostponed* q_postponed) {
      double distance_sq_lo =
          q_node.bound().MinDistanceSq(r_node.bound());
      bool need_expansion;
      
      DEBUG_MSG(1.0, "tkde: ConsiderPairIntrinsic");
      
      if (distance_sq_lo > param.kernel.bandwidth_sq()) {
        DEBUG_MSG(1.0, "tkde: Exclusion");
        need_expansion = false;
      } else {
        double distance_sq_hi =
            q_node.bound().MaxDistanceSq(r_node.bound());

        if (0) {
        // REMOVED because this was causing bugs.
        // Someone please figure out what's wrong with the moment expansion!
        
        /*distance_sq_hi < param.kernel.bandwidth_sq()) {
          DEBUG_MSG(1.0, "tkde: Inclusion");
          q_postponed->moment_info.Add(r_node.stat().moment_info);
          need_expansion = false;*/
        } else {
          DEBUG_MSG(1.0, "tkde: Overlap - need explore");
          //delta->d_density = r_node.stat().moment_info.ComputeKernelSumRange(
          //    param, q_node.bound());
          //max(delta->d_density.lo, 0.0);
          // this method seemed like a good idea, but the upper bound it
          // computes is unfortunately bogus
          delta->d_density.lo = 0;
          delta->d_density.hi = r_node.count() *
              param.kernel.EvalUnnormOnSq(distance_sq_lo);
          need_expansion = true;
        }
      }

      return need_expansion;
    }

    static bool ConsiderPairExtrinsic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        const Delta& delta,
        const QSummaryResult& q_summary_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
    }

    static bool ConsiderQueryTermination(
        const Param& param,
        const QNode& q_node,
        const QSummaryResult& q_summary_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      bool need_expansion = false;

      DEBUG_ASSERT(q_summary_result.density.lo < q_summary_result.density.hi);

      if (unlikely(q_summary_result.label != LAB_EITHER)) {
        DEBUG_ASSERT((q_summary_result.label & q_postponed->label) != LAB_NEITHER);
        q_postponed->label = q_summary_result.label;
      } else if (unlikely(q_summary_result.density.lo > param.thresh.hi)) {
        q_postponed->label = LAB_HI;
      } else if (unlikely(q_summary_result.density.hi < param.thresh.lo)) {
        q_postponed->label = LAB_LO;
      } else {
        need_expansion = true;
      }

      return need_expansion;
    }

    /**
     * Computes a heuristic for how early a computation should occur -- smaller
     * values are earlier.
     */
    static double Heuristic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        const Delta& delta) {
      return r_node.bound().MinToMidSq(q_node.bound());
    }
  };
};

void TkdeMain(datanode *module) {
  thor::MonochromaticDualTreeMain<Tkde, DualTreeDepthFirst<Tkde> >(
      module, "tkde");
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  TkdeMain(fx_root);
  
  fx_done();
}

