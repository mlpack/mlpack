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
    index_t index_;
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
     * Gets the index.
     */
    index_t index() const {
      return index_;
    }
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
    void Set(const Param& param, index_t index, const Vector& data) {
      index_ = index;
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
     * Normalization constat for positive points. Similar for _neg.
     *
     * Hi and lo also acount for threshold (coincidentally, from the
     * other class' point of view).  Example use:
     *
     *   const_pos.lo * density_pos.lo * pi_pos.lo
     *   > const_neg.hi * density_neg.hi * pi_neg.hi
     */
    DRange const_pos;
    DRange const_neg;
    /** Like const_pos but leaving out one point. Similar for _neg. */
    DRange const_pos_loo;
    DRange const_neg_loo;
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
    /** The specified threshold for certainty of positive class. */
    double threshold;
    /** Peak kernel values (to eliminate recomputation). */
    double peak_neg;
    double peak_pos;
    /** Whether to compute densities leaving out matching indices. */
    bool loo;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(const_pos);
      OT_MY_OBJECT(const_neg);
      OT_MY_OBJECT(const_pos_loo);
      OT_MY_OBJECT(const_neg_loo);
      OT_MY_OBJECT(kernel_pos);
      OT_MY_OBJECT(kernel_neg);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(count_all);
      OT_MY_OBJECT(count_pos);
      OT_MY_OBJECT(count_neg);
      OT_MY_OBJECT(threshold);
      OT_MY_OBJECT(peak_pos);
      OT_MY_OBJECT(peak_neg);
      OT_MY_OBJECT(loo);
    }

   public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      kernel_pos.Init(fx_param_double_req(module, "h_pos"));
      kernel_neg.Init(fx_param_double_req(module, "h_neg"));
      threshold = fx_param_double(module, "threshold", 0.5);
      loo = !fx_param_exists(module, "q");
    }

    /**
     * Reflect some aspects of the data (Req THOR).
     *
     * Note: called *after* reading all data.
     */
    void SetDimensions(index_t vector_dimension, index_t n_points) {
      dim = vector_dimension; // last two cols already trimmed
      count_all = n_points;
      peak_pos = kernel_pos.EvalUnnormOnSq(0)
	/ kernel_pos.CalcNormConstant(dim);
      peak_neg = kernel_neg.EvalUnnormOnSq(0)
	/ kernel_neg.CalcNormConstant(dim);
    }

    /**
     * Finalize parameters (Not THOR).
     */
    void ComputeConsts(int count_pos_in, int count_neg_in) {
      double epsilon = min(threshold, 1 - threshold) * 1e-3;

      count_pos = count_pos_in;
      count_neg = count_neg_in;

      double norm_pos = kernel_pos.CalcNormConstant(dim) * count_pos;
      const_pos.lo = (1 - threshold - epsilon) / norm_pos;
      const_pos.hi = (1 - threshold + epsilon) / norm_pos;

      double norm_neg = kernel_neg.CalcNormConstant(dim) * count_neg;
      const_neg.lo = (threshold - epsilon) / norm_neg;
      const_neg.hi = (threshold + epsilon) / norm_neg;

      norm_pos = kernel_pos.CalcNormConstant(dim) * (count_pos - 1);
      const_pos_loo.lo = (1 - threshold - epsilon) / norm_pos;
      const_pos_loo.hi = (1 - threshold + epsilon) / norm_pos;

      norm_neg = kernel_neg.CalcNormConstant(dim) * (count_neg - 1);
      const_neg_loo.lo = (threshold - epsilon) / norm_neg;
      const_neg_loo.hi = (threshold + epsilon) / norm_neg;

      ot::Print(dim);
      ot::Print(count_all);

      ot::Print(count_pos);
      ot::Print(kernel_pos);
      ot::Print(const_pos);

      ot::Print(count_neg);
      ot::Print(kernel_neg);
      ot::Print(const_neg);
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

    /**
     * Compute kernel sum for a region of reference points assuming we have
     * the actual query point.
     */
    double ComputeKernelSum(const Kernel& kernel, const Vector& q) const {
      double quadratic_term =
          + count * la::Dot(q, q)
          - 2.0 * la::Dot(q, mass)
          + sumsq;
      return count - quadratic_term * kernel.inv_bandwidth_sq();
    }

    double ComputeKernelSum(const Kernel& kernel, double distance_squared,
        double center_dot_center) const {
      //q*q - 2qr + rsumsq
      //q*q - 2qr + r*r - r*r
      double quadratic_term =
          (distance_squared - center_dot_center) * count
          + sumsq;

      return -quadratic_term * kernel.inv_bandwidth_sq() + count;
    }

    DRange ComputeKernelSumRange(const Kernel& kernel,
        const Bound& query_bound) const {
      DRange density_bound;
      Vector center;
      double center_dot_center = la::Dot(mass, mass) / count / count;
      
      DEBUG_ASSERT(count != 0);

      center.Copy(mass);
      la::Scale(1.0 / count, &center);

      density_bound.lo = ComputeKernelSum(kernel,
          query_bound.MaxDistanceSq(center), center_dot_center);
      density_bound.hi = ComputeKernelSum(kernel,
          query_bound.MinDistanceSq(center), center_dot_center);

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
    /** Bounds for query self-contrib (undone in LOO case). */
    DRange loo_contrib;

    OT_DEF_BASIC(NbcStat) {
      OT_MY_OBJECT(moment_info_pos);
      OT_MY_OBJECT(moment_info_neg);
      OT_MY_OBJECT(bound_pos);
      OT_MY_OBJECT(bound_neg);
      OT_MY_OBJECT(count_pos);
      OT_MY_OBJECT(count_neg);
      OT_MY_OBJECT(pi_pos);
      OT_MY_OBJECT(pi_neg);
      OT_MY_OBJECT(loo_contrib);
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
      if (param.loo) {
	loo_contrib.InitEmptySet();
      } else {
	loo_contrib.Init(0, 0);
      }
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
      pi_pos |= point.pi_pos();
      pi_neg |= point.pi_neg();
      if (param.loo) {
	if (point.is_pos()) {
	  loo_contrib |= param.peak_pos;
	} else {
	  loo_contrib |= param.peak_neg;
	}
      }
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
      pi_pos |= stat.pi_pos;
      pi_neg |= stat.pi_neg;
      loo_contrib |= stat.loo_contrib;
    }

    /**
     * Finish accumulating data; for instance, for mean, divide by the
     * number of points.
     */
    void Postprocess(const Param& param, const Bound& bound, index_t n) {
    }
  };
  typedef NbcStat RStat;
  typedef NbcStat QStat;
 
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
    MomentInfo moment_info_pos;
    MomentInfo moment_info_neg;
    /** We pruned an entire part of the tree with a particular label. */
    int label;

    OT_DEF_BASIC(QPostponed) {
      OT_MY_OBJECT(moment_info_pos);
      OT_MY_OBJECT(moment_info_neg);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      moment_info_pos.Init(param);
      moment_info_neg.Init(param);
      label = LAB_EITHER;
    }

    void Reset(const Param& param) {
      moment_info_pos.Reset();
      moment_info_neg.Reset();
      label = LAB_EITHER;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      label &= other.label;
      DEBUG_ASSERT_MSG(label != LAB_NEITHER, "Conflicting labels?");
      moment_info_pos.Add(other.moment_info_pos);
      moment_info_neg.Add(other.moment_info_neg);
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
      if (param.loo) {
	// Withhold contribution of q from density for its class
	if (q.is_pos()) {
	  if (param.const_pos_loo.lo * (density_pos - param.peak_pos) * q.pi_pos()
	      > param.const_neg.hi * density_neg * q.pi_neg()) {
	    label &= LAB_POS;
	  } else if (param.const_neg.lo * density_neg * q.pi_neg()
	      > param.const_pos_loo.hi * (density_pos - param.peak_pos) * q.pi_pos()) {
	    label &= LAB_NEG;
	  }
	} else {
	  if (param.const_pos.lo * density_pos * q.pi_pos()
	      > param.const_neg_loo.hi * (density_neg - param.peak_neg) * q.pi_neg()) {
	    label &= LAB_POS;
	  } else if (param.const_neg_loo.lo * (density_neg - param.peak_neg) * q.pi_neg()
	      > param.const_pos.hi * density_pos * q.pi_pos()) {
	    label &= LAB_NEG;
	  }
	}
      } else {
	if (param.const_pos.lo * density_pos * q.pi_pos()
	    > param.const_neg.hi * density_neg * q.pi_neg()) {
	  label &= LAB_POS;
	} else if (param.const_neg.lo * density_neg * q.pi_neg()
	    > param.const_pos.hi * density_pos * q.pi_pos()) {
	  label &= LAB_NEG;
	}
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
            param.kernel_pos, q.vec());
      }
      if (!postponed.moment_info_neg.is_empty()) {
        density_neg += postponed.moment_info_neg.ComputeKernelSum(
            param.kernel_neg, q.vec());
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
            param.kernel_pos, q_node.bound());
        change_made = true;
      }
      if (unlikely(!postponed.moment_info_neg.is_empty())) {
        density_neg += postponed.moment_info_neg.ComputeKernelSumRange(
            param.kernel_neg, q_node.bound());
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
          double(count_pos) / param.count_all * 100.0);
      fx_format_result(datanode, "count_unknown", "%"LI"d",
          count_unknown);
      fx_format_result(datanode, "percent_unknown", "%.05f",
          double(count_unknown) / param.count_all * 100.0);
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
#ifdef CHEC_POS_NEG_BOUNDS
    bool do_pos;
    bool do_neg;
#endif

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

#ifdef CHECK_POS_NEG_BOUNDS
      if (unlikely(
	   (r_node.stat().count_pos == 0 ||
	    r_node.stat().bound_pos.MinDistanceSq(q.vec())
	    > param.kernel_pos.bandwidth_sq()) &&
	   (r_node.stat().count_neg == 0 ||
	    r_node.stat().bound_neg.MinDistanceSq(q.vec())
	    > param.kernel_neg.bandwidth_sq()))) {
	return false;
      }

      do_pos = true;
      do_neg = true;
      if (r_node.stat().count_pos > 0) {
	if (unlikely(
	     r_node.stat().bound_pos.MaxDistanceSq(q.vec())
	     < param.kernel_pos.bandwidth_sq())) {
	  q_result->density_pos +=
	    r_node.stat().moment_info_pos.ComputeKernelSum(
		param.kernel_pos, q.vec());
	  do_pos = false;
	}
      }
      if (r_node.stat().count_neg > 0) {
	if (unlikely(
	     r_node.stat().bound_neg.MaxDistanceSq(q.vec())
	     < param.kernel_neg.bandwidth_sq())) {
	  q_result->density_neg +=
	    r_node.stat().moment_info_neg.ComputeKernelSum(
		param.kernel_neg, q.vec());
	  do_neg = false;
	}
      }

      density_pos = 0.0;
      density_neg = 0.0;

      return do_pos || do_neg;
#else
      if (unlikely(
	    r_node.bound().MinDistanceSq(q.vec())
	    > max(param.kernel_pos.bandwidth_sq(),
		  param.kernel_neg.bandwidth_sq()))) {
	return false;
      }

      if (unlikely(
	   r_node.bound().MaxDistanceSq(q.vec())
	   < min(param.kernel_pos.bandwidth_sq(),
		 param.kernel_neg.bandwidth_sq()))) {
	if (r_node.stat().count_pos > 0) {
	  q_result->density_pos +=
	    r_node.stat().moment_info_pos.ComputeKernelSum(
	        param.kernel_pos, q.vec());
	}
	if (r_node.stat().count_neg > 0) {
	  q_result->density_neg +=
	    r_node.stat().moment_info_neg.ComputeKernelSum(
	        param.kernel_neg, q.vec());
	}
	return false;
      }

      density_pos = 0.0;
      density_neg = 0.0;

      return true;
#endif
    }

    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
#ifdef CHECK_POS_NEG_BOUNDS
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
#else
      double distance = la::DistanceSqEuclidean(q.vec(), r.vec());
      if (r.is_pos()) {
	density_pos += param.kernel_pos.EvalUnnormOnSq(distance);
      } else {
	density_neg += param.kernel_neg.EvalUnnormOnSq(distance);
      }
#endif
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

      if (param.loo) {
	// Withhold contribution of q from density for its class
	if (q.is_pos()) {
	  if (unlikely(
	       param.const_pos_loo.lo * (total_density_pos.lo - param.peak_pos) * q.pi_pos()
	       > param.const_neg.hi * total_density_neg.hi * q.pi_neg())) {
	    q_result->label &= LAB_POS;
	  } else if (unlikely(
	       param.const_neg.lo * total_density_neg.lo * q.pi_neg()
	       > param.const_pos_loo.hi * (total_density_pos.hi - param.peak_pos) * q.pi_pos())) {
	    q_result->label &= LAB_NEG;
	  }
	} else {
	  if (unlikely(
	       param.const_pos.lo * total_density_pos.lo * q.pi_pos()
	       > param.const_neg_loo.hi * (total_density_neg.hi - param.peak_neg) * q.pi_neg())) {
	    q_result->label &= LAB_POS;
	  } else if (unlikely(
	       param.const_neg_loo.lo * (total_density_neg.lo - param.peak_neg) * q.pi_neg()
	       > param.const_pos.hi * total_density_pos.hi * q.pi_pos())) {
	    q_result->label &= LAB_NEG;
	  }
	}
      } else {
	if (unlikely(
	     param.const_pos.lo * total_density_pos.lo * q.pi_pos()
	     > param.const_neg.hi * total_density_neg.hi * q.pi_neg())) {
	  q_result->label &= LAB_POS;
	} else if (unlikely(
	     param.const_neg.lo * total_density_neg.lo * q.pi_neg()
	     > param.const_pos.hi * total_density_pos.hi * q.pi_pos())) {
	  q_result->label &= LAB_NEG;
	}
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
      DEBUG_MSG(1.0, "nbc: ConsiderPairIntrinsic");

      double d_density_pos_hi = 0;
      if (r_node.stat().count_pos > 0) {
	double distance_sq_pos_lo =
	  r_node.stat().bound_pos.MinDistanceSq(q_node.bound());
	d_density_pos_hi =
	  param.kernel_pos.EvalUnnormOnSq(distance_sq_pos_lo);
      }

      double d_density_neg_hi = 0;
      if (r_node.stat().count_neg > 0) {
	double distance_sq_neg_lo =
	  r_node.stat().bound_neg.MinDistanceSq(q_node.bound());
	d_density_neg_hi =
	  param.kernel_neg.EvalUnnormOnSq(distance_sq_neg_lo);
      }

      if (d_density_pos_hi == 0 && d_density_neg_hi == 0) {
        DEBUG_MSG(1.0, "nbc: Exclusion");
	return false;
      }

#ifdef CHECK_POS_NEG_BOUNDS
      if ((r_node.stat().count_pos == 0 ||
	   r_node.stat().bound_pos.MaxDistanceSq(q_node.bound())
	   < param.kernel_pos.bandwidth_sq()) &&
	  (r_node.stat().count_neg == 0 ||
	   r_node.stat().bound_neg.MaxDistanceSq(q_node.bound())
	   < param.kernel_neg.bandwidth_sq())) {
	if (r_node.stat().count_pos > 0) {
	  q_postponed->moment_info_pos.Add(r_node.stat().moment_info_pos);
	}
	if (r_node.stat().count_neg > 0) {
	  q_postponed->moment_info_neg.Add(r_node.stat().moment_info_neg);
	}
        DEBUG_MSG(1.0, "nbc: Inclusion");
	return false;
      }
#else
      if (r_node.bound().MaxDistanceSq(q_node.bound())
	  < param.kernel_pos.bandwidth_sq()) {
	if (r_node.stat().count_pos > 0) {
	  q_postponed->moment_info_pos.Add(r_node.stat().moment_info_pos);
	}
	if (r_node.stat().count_neg > 0) {
	  q_postponed->moment_info_neg.Add(r_node.stat().moment_info_neg);
	}
        DEBUG_MSG(1.0, "nbc: Inclusion");
	return false;
      }
#endif

      delta->d_density_pos.lo = 0;
      delta->d_density_pos.hi = r_node.stat().count_pos * d_density_pos_hi;
      delta->d_density_neg.lo = 0;
      delta->d_density_neg.hi = r_node.stat().count_neg * d_density_neg_hi;
      
      return true;
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
      DEBUG_ASSERT(q_summary_result.density_pos.lo <= q_summary_result.density_pos.hi);
      DEBUG_ASSERT(q_summary_result.density_neg.lo <= q_summary_result.density_neg.hi);

      if (unlikely(q_summary_result.label != LAB_EITHER)) {
        DEBUG_ASSERT((q_summary_result.label & q_postponed->label) != LAB_NEITHER);
        q_postponed->label = q_summary_result.label;
	return false;
      }

      if (param.loo) {
#ifdef CHECK_POS_NEG_LOO_BOUNDS
	// Make sure bounds hold for both pos and neg queries
	if (unlikely(
	     (q_node.stat().count_pos == 0 ||
	       param.const_pos_loo.lo
	         * (q_summary_result.density_pos.lo - param.peak_pos)
	         * q_node.stat().pi_pos.lo
	       > param.const_neg.hi
	         * q_summary_result.density_neg.hi
	         * q_node.stat().pi_neg.hi)
	     && (q_node.stat().count_neg == 0 ||
	       param.const_pos.lo
	         * q_summary_result.density_pos.lo
	         * q_node.stat().pi_pos.lo
	       > param.const_neg_loo.hi
	         * (q_summary_result.density_neg.hi - param.peak_neg)
	         * q_node.stat().pi_neg.hi))) {
	  q_postponed->label = LAB_POS;
	  return false;
	} else if (unlikely(
	     (q_node.stat().count_pos == 0 ||
	       param.const_neg.lo
	         * q_summary_result.density_neg.lo
	         * q_node.stat().pi_neg.lo
	       > param.const_pos_loo.hi
	         * (q_summary_result.density_pos.hi - param.peak_pos)
	         * q_node.stat().pi_pos.hi)
	     && (q_node.stat().count_neg == 0 ||
	       param.const_neg_loo.lo
	         * (q_summary_result.density_neg.lo - param.peak_neg)
	         * q_node.stat().pi_neg.lo
	       > param.const_pos.hi
	         * q_summary_result.density_pos.hi
	         * q_node.stat().pi_pos.hi))) {
	  q_postponed->label = LAB_NEG;
	  return false;
	}
#else
	// Note const_pos.lo < const_pos_loo.lo, etc.
	if (unlikely(
	     param.const_pos.lo
	       * (q_summary_result.density_pos.lo - param.peak_pos)
	       * q_node.stat().pi_pos.lo
	     > param.const_neg_loo.hi
	       * q_summary_result.density_neg.hi
	       * q_node.stat().pi_neg.hi)) {
	  q_postponed->label = LAB_POS;
	  return false;
	} else if (unlikely(
	     param.const_neg.lo
	       * (q_summary_result.density_neg.lo - param.peak_neg)
	       * q_node.stat().pi_neg.lo
	     > param.const_pos_loo.hi
	       * q_summary_result.density_pos.hi
	       * q_node.stat().pi_pos.hi)) {
	  q_postponed->label = LAB_NEG;
	  return false;
	}
#endif
      } else {
	if (unlikely(
	     param.const_pos.lo
	       * q_summary_result.density_pos.lo
	       * q_node.stat().pi_pos.lo
	     > param.const_neg.hi
	       * q_summary_result.density_neg.hi
	       * q_node.stat().pi_neg.hi)) {
	  q_postponed->label = LAB_POS;
	  return false;
	} else if (unlikely(
	     param.const_neg.lo
	       * q_summary_result.density_neg.lo
	       * q_node.stat().pi_neg.lo
	     > param.const_pos.hi
	       * q_summary_result.density_pos.hi
	       * q_node.stat().pi_pos.hi)) {
	  q_postponed->label = LAB_NEG;
	  return false;
	}
      }

      return true;
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

void NbcMain(datanode *module) {
  // TODO: LOO, reporting, multi-bw, recursive "bfs", multi-thresh
  //   make sure bichromatic works
  //   for LOO: easiest to include self but correct when pruning
  //     i.e. subtract lb self-contrib from ub dens; vice versa
  //     make sure to use correct coefficients
  //     lb dens - ub self-contrib should be clamped positive
  //     lb/ub self-contrib deps on bws of pos/neg, what's in node

  //thor::MonochromaticDualTreeMain<Tkde, DualTreeDepthFirst<Tkde> >(
  //    module, "tkde");
  const char *gnp_name = "nbc";
  const int DATA_CHANNEL = 110;
  const int Q_RESULTS_CHANNEL = 120;
  const int GNP_CHANNEL = 200;
  double results_megs = fx_param_double(module, "results/megs", 1000);
  DistributedCache *q_points_cache;
  DistributedCache *r_points_cache;
  index_t n_q_points;
  index_t n_r_points;
  ThorTree<Nbc::Param, Nbc::QPoint, Nbc::QNode> *q_tree;
  ThorTree<Nbc::Param, Nbc::RPoint, Nbc::RNode> *r_tree;
  DistributedCache q_results;
  Nbc::Param param;

  rpc::Init();

  if (!rpc::is_root()) {
    fx_silence();
  }

  fx_submodule(module, NULL, "io"); // influnce output order

  param.Init(fx_submodule(module, gnp_name, gnp_name));

  fx_timer_start(module, "read");
  r_points_cache = new DistributedCache();
  n_r_points = thor::ReadPoints<Nbc::RPoint>(
      param, DATA_CHANNEL + 0, DATA_CHANNEL + 1,
      fx_submodule(module, "r", "r"),
      r_points_cache);
  if (fx_param_exists(module, "q")) {
    q_points_cache = new DistributedCache();
    n_q_points = thor::ReadPoints<Nbc::QPoint>(
        param, DATA_CHANNEL + 2, DATA_CHANNEL + 3,
	fx_submodule(module, "q", "q"),
	q_points_cache);
  } else {
    q_points_cache = r_points_cache;
    n_q_points = n_r_points;
  }
  fx_timer_stop(module, "read");

  Nbc::RPoint default_point;
  CacheArray<Nbc::RPoint>::GetDefaultElement(
      r_points_cache, &default_point);
  param.SetDimensions(default_point.vec().length(), n_r_points);

  fx_timer_start(module, "tree");
  r_tree = new ThorTree<Nbc::Param, Nbc::RPoint, Nbc::RNode>();
  thor::CreateKdTree<Nbc::RPoint, Nbc::RNode>(
      param, DATA_CHANNEL + 4, DATA_CHANNEL + 5,
      fx_submodule(module, "r_tree", "r_tree"),
      n_r_points, r_points_cache, r_tree);
  if (fx_param_exists(module, "q")) {
    q_tree = new ThorTree<Nbc::Param, Nbc::QPoint, Nbc::QNode>();
    thor::CreateKdTree<Nbc::QPoint, Nbc::QNode>(
        param, DATA_CHANNEL + 6, DATA_CHANNEL + 7,
	fx_submodule(module, "q_tree", "q_tree"),
	n_q_points, q_points_cache, q_tree);
  } else {
    q_tree = r_tree;
  }
  fx_timer_stop(module, "tree");

  // This should have been a first-order reduce at the time of read
  param.ComputeConsts(r_tree->root().stat().count_pos,
		      r_tree->root().stat().count_neg);

  Nbc::QResult default_result;
  default_result.Init(param);
  q_tree->CreateResultCache(Q_RESULTS_CHANNEL, default_result,
			    results_megs, &q_results);

  Nbc::GlobalResult *global_result;
  thor::RpcDualTree<Nbc, DualTreeDepthFirst<Nbc> >(
      fx_submodule(module, "gnp", "gnp"), GNP_CHANNEL, param,
      q_tree, r_tree, &q_results, &global_result);
  delete global_result;

  // Emit the results; this needs to be folded into THOR
  Matrix classifications;
  classifications.Init(1, n_q_points);
  if (rpc::is_root()) {
    CacheArray<Nbc::QResult> result_array;
    CacheArray<Nbc::QPoint> points_array;
    result_array.Init(&q_results, BlockDevice::M_READ);
    points_array.Init(q_points_cache, BlockDevice::M_READ);
    CacheReadIter<Nbc::QResult> result_iter(&result_array, 0);
    CacheReadIter<Nbc::QPoint> points_iter(&points_array, 0);
    for (index_t i = 0; i < n_q_points; i++,
	   result_iter.Next(), points_iter.Next()) {
      classifications.set(0, (*points_iter).index(), (*result_iter).label);
    }
  }
  data::Save(fx_param_str(module, "out", "out.csv"), classifications);

  rpc::Done();
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  NbcMain(fx_root);
  
  fx_done();
}
