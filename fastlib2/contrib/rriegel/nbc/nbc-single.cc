#include "fastlib/fastlib_int.h"
#include "fastlib/thor/thor.h"

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
      vec_.Init(schema.n_features() - !param.no_priors - !param.no_labels);
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
      if (param.no_labels) {
	is_pos_ = false;
      } else {
	is_pos_ = (data[data.length() - !param.no_priors - 1] != 0.0);
      }
      if (param.no_priors || param.prior_override >= 0) {
	pi_ = param.prior_override;
      } else {
	pi_ = data[data.length() - 1];
      }
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
    /** The kernel for positive points. Similar for _neg. */
    Kernel kernel_pos;
    Kernel kernel_neg;
    /**
     * Normalization coeff for positive points. Similar for _neg.
     *
     * _loo versions should be used to classify a point as its own
     * class in the LOO case.  Also in the LOO case, the non _loo
     * version is corrected to classify out of class; this coeff is
     * necessarily smaller than the _loo version, and thus suitable
     * for all lower bounds.  Example use:
     *
     *   coeff_pos * density_pos.lo * pi_pos.lo
     *     > density_neg.hi * pi_neg.hi
     *
     * is tight for classifying a negative point as positive and won't
     * misclassify positive points.
     */
    double coeff_pos;
    double coeff_neg;
    double coeff_pos_loo;
    double coeff_neg_loo;
    /** The dimensionality of the data sets. */
    index_t dim;
    /** Number of reference points. */
    index_t count_all;
    /** Number of positive reference points. Similar for _neg. */
    index_t count_pos;
    index_t count_neg;
    /** The specified threshold for certainty of positive class. */
    double threshold;
    /** If set, will be used as prior for all data. */
    double prior_override;
    /** Peak kernel values. */
    double peak_neg;
    double peak_pos;
    /** Min and max bandwidth values (to eliminate recomputation). */
    double max_bandwidth_sq;
    double min_bandwidth_sq;
    /** Whether to compute densities leaving out matching indices. */
    bool loo;
    /** Whether to read labels from current set. Similar for _priors. */
    bool no_labels;
    bool no_priors;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(kernel_pos);
      OT_MY_OBJECT(kernel_neg);
      OT_MY_OBJECT(coeff_pos);
      OT_MY_OBJECT(coeff_neg);
      OT_MY_OBJECT(coeff_pos_loo);
      OT_MY_OBJECT(coeff_neg_loo);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(count_all);
      OT_MY_OBJECT(count_pos);
      OT_MY_OBJECT(count_neg);
      OT_MY_OBJECT(threshold);
      OT_MY_OBJECT(prior_override);
      OT_MY_OBJECT(peak_pos);
      OT_MY_OBJECT(peak_neg);
      OT_MY_OBJECT(max_bandwidth_sq);
      OT_MY_OBJECT(min_bandwidth_sq);
      OT_MY_OBJECT(loo);
      OT_MY_OBJECT(no_labels);
      OT_MY_OBJECT(no_priors);
    }

   public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      kernel_pos.Init(fx_param_double_req(module, "h_pos"));
      kernel_neg.Init(fx_param_double_req(module, "h_neg"));
      threshold = fx_param_double(module, "threshold", 0.5);
      prior_override = fx_param_double(module, "prior", -1.0);
      max_bandwidth_sq =
	max(kernel_pos.bandwidth_sq(),kernel_neg.bandwidth_sq());
      min_bandwidth_sq =
	min(kernel_pos.bandwidth_sq(),kernel_neg.bandwidth_sq());
      loo = false;
    }

    /**
     * Reflect some aspects of the data (Req THOR).
     *
     * Note: called *after* reading all data.
     */
    void SetDimensions(index_t vector_dimension, index_t n_points) {
      dim = vector_dimension; // last two cols already trimmed
      count_all = n_points;
      peak_pos = kernel_pos.EvalUnnormOnSq(0);
      peak_neg = kernel_neg.EvalUnnormOnSq(0);
    }

    /**
     * Finalize parameters (Not THOR).
     */
    void ComputeConsts(int count_pos_in, int count_neg_in) {
      count_pos = count_pos_in;
      count_neg = count_neg_in;

      /* The logic behind the below is semi-involved.
       *
       * First, observe we must divide kernel values by their
       * normalizing constant.  Second, we must divide sums of kernel
       * values by the number of points summed over (withholding one
       * appropriately in the LOO case).  Third, we multiply by the
       * threshold or its compliment (offset by epsilon for stability)
       * in order to test that positively classified points have the
       * desired confidence.  So, that's:
       *
       *   (1 - threshold - epsilon) / (norm_pos * count_pos)
       *
       * for a lower-bound value for positive densities.  We'll be
       * comparing this against upper-bound values for negative
       * densities, or:
       *
       *   (threshold + epsilon) / (norm_neg * count_neg_loo)
       *
       * where count_neg_loo is less one point in the LOO case.  These
       * are necessarily positive and normally live on either side of
       * greater-than sign, but to avoid one multiplication, we form
       * the ratio, as computed below.
       */

      double epsilon = min(threshold, 1 - threshold) * 1e-6;
      double norm_pos = kernel_pos.CalcNormConstant(dim);
      double norm_neg = kernel_neg.CalcNormConstant(dim);

      index_t count_pos_loo = loo ? count_pos - 1 : count_pos;
      index_t count_neg_loo = loo ? count_neg - 1 : count_neg;

      coeff_pos = (1 - threshold - epsilon) * norm_neg * count_neg_loo
	/ ((threshold + epsilon) * norm_pos * count_pos);
      coeff_neg = (threshold - epsilon) * norm_pos * count_pos_loo
	/ ((1 - threshold + epsilon) * norm_neg * count_neg);

      coeff_pos_loo = (1 - threshold - epsilon) * norm_neg * count_neg
	/ ((threshold + epsilon) * norm_pos * count_pos_loo);
      coeff_neg_loo = (threshold - epsilon) * norm_pos * count_pos
	/ ((1 - threshold + epsilon) * norm_neg * count_neg_loo);

      ot::Print(loo);

      ot::Print(dim);
      ot::Print(count_all);

      ot::Print(count_pos);
      ot::Print(kernel_pos);
      ot::Print(coeff_pos);
      ot::Print(coeff_pos_loo);

      ot::Print(count_neg);
      ot::Print(kernel_neg);
      ot::Print(coeff_neg);
      ot::Print(coeff_neg_loo);
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
    /** Number of positive points. Similar for _neg. */
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
      OT_MY_OBJECT(pi_pos);
      OT_MY_OBJECT(pi_neg);
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
      pi_pos |= point.pi_pos();
      pi_neg |= point.pi_neg();
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

    void Seed(const Param& param, const QPoint& q) {
      if (param.loo) {
	if (q.is_pos()) {
	  density_pos -= param.peak_pos;
	} else {
	  density_neg -= param.peak_neg;
	}
      }
    }

    // Why does this have a q_index and r_root?  RR
    void Postprocess(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_root) {
      if (label == LAB_EITHER) {
	if (param.loo) {
	  // Use proper coeffs for q's class
	  if (q.is_pos()) {
	    if (param.coeff_pos_loo * density_pos * q.pi_pos()
		> density_neg * q.pi_neg()) {
	      label &= LAB_POS;
	    } else if (param.coeff_neg * density_neg * q.pi_neg()
		> density_pos * q.pi_pos()) {
	      label &= LAB_NEG;
	    }
	  } else {
	    if (param.coeff_pos * density_pos * q.pi_pos()
		> density_neg * q.pi_neg()) {
	      label &= LAB_POS;
	    } else if (param.coeff_neg_loo * density_neg * q.pi_neg()
		> density_pos * q.pi_pos()) {
	      label &= LAB_NEG;
	    }
	  }
	} else {
	  if (param.coeff_pos * density_pos * q.pi_pos()
	      > density_neg * q.pi_neg()) {
	    label &= LAB_POS;
	  } else if (param.coeff_neg * density_neg * q.pi_neg()
	      > density_pos * q.pi_pos()) {
	    label &= LAB_NEG;
	  }
	}
	DEBUG_ASSERT_MSG(label != LAB_NEITHER,
	    "Conflicting labels: [%g, %g]; %g > %g; %g > %g",
	    density_pos, density_neg,
	    param.coeff_pos * density_pos * q.pi_pos(),
	    density_neg * q.pi_neg(),
	    param.coeff_neg * density_neg * q.pi_neg(),
	    density_pos * q.pi_pos());
      }
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

    void Seed(const Param& param, const QNode& q_node) {
      if (param.loo) {
	if (q_node.stat().count_pos > 0) {
	  density_pos.lo -= param.peak_pos;
	} else {
	  density_neg.hi -= param.peak_neg;
	}
	if (q_node.stat().count_neg > 0) {
	  density_neg.lo -= param.peak_neg;
	} else {
	  density_pos.hi -= param.peak_pos;
	}
      }
    }

    // Why does this have a q_node?  RR
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

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      label &= postponed.label;
      DEBUG_ASSERT(label != LAB_NEITHER);

      if (unlikely(!postponed.moment_info_pos.is_empty())) {
        density_pos += postponed.moment_info_pos.ComputeKernelSumRange(
            param.kernel_pos, q_node.bound());
      }
      if (unlikely(!postponed.moment_info_neg.is_empty())) {
        density_neg += postponed.moment_info_neg.ComputeKernelSumRange(
            param.kernel_neg, q_node.bound());
      }
    }
  };

  /**
   * A simple postprocess-step global result.
   */
  struct GlobalResult {
   public:
    index_t count_pos;
    index_t count_neg;
    index_t count_unknown;
    
    OT_DEF_BASIC(GlobalResult) {
      OT_MY_OBJECT(count_pos);
      OT_MY_OBJECT(count_neg);
      OT_MY_OBJECT(count_unknown);
    }

   public:
    void Init(const Param& param) {
      count_pos = 0;
      count_neg = 0;
      count_unknown = 0;
    }
    void Accumulate(const Param& param, const GlobalResult& other) {
      count_pos += other.count_pos;
      count_neg += other.count_neg;
      count_unknown += other.count_unknown;
    }
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}
    void Postprocess(const Param& param) {}
    void Report(const Param& param, datanode *datanode) {
      index_t total = count_pos + count_neg + count_unknown;

      fx_format_result(datanode, "count_pos", "%"LI"d",
          count_pos);
      fx_format_result(datanode, "percent_pos", "%.05f",
          double(count_pos) / total * 100.0);
      fx_format_result(datanode, "count_neg", "%"LI"d",
          count_neg);
      fx_format_result(datanode, "percent_neg", "%.05f",
          double(count_neg) / total * 100.0);
      fx_format_result(datanode, "count_unknown", "%"LI"d",
          count_unknown);
      fx_format_result(datanode, "percent_unknown", "%.05f",
          double(count_unknown) / total * 100.0);
    }
    void ApplyResult(const Param& param,
        const QPoint& q_point, index_t q_i,
        const QResult& result) {
      if (result.label == LAB_POS) {
        ++count_pos;
      } else if (result.label == LAB_NEG) {
        ++count_neg;
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

   public:
    void Init(const Param& param) {}

    // notes
    // - this function must assume that global_result is incomplete (which is
    // reasonable in allnn)
    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_node,
	const Delta& delta,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
      if (unlikely(q_result->label != LAB_EITHER)) {
        return false;
      }

      // Tighter check possible against pos and neg separately; slow

      DRange dist_sq = r_node.bound().RangeDistanceSq(q.vec());

      if (unlikely(dist_sq.lo > param.max_bandwidth_sq)) {
	return false;
      }

      if (unlikely(dist_sq.hi < param.min_bandwidth_sq)) {
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
    }

    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
      double distance = la::DistanceSqEuclidean(q.vec(), r.vec());
      if (r.is_pos()) {
	density_pos += param.kernel_pos.EvalUnnormOnSq(distance);
      } else {
	density_neg += param.kernel_neg.EvalUnnormOnSq(distance);
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

      // const_pos.lo <= const_pos_loo.lo, etc.; if !param.loo, ==
      // - tighter check possible given q's class, but effect is just tiny
      if (unlikely(
	   param.coeff_pos * total_density_pos.lo * q.pi_pos()
	   > total_density_neg.hi * q.pi_neg())) {
	q_result->label &= LAB_POS;
      } else if (unlikely(
	   param.coeff_neg * total_density_neg.lo * q.pi_neg()
	   > total_density_pos.hi * q.pi_pos())) {
	q_result->label &= LAB_NEG;
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
	const Delta& parent_delta,
        Delta* delta,
        GlobalResult* global_result,
        QPostponed* q_postponed) {
      VERBOSE_MSG(1.0, "nbc: ConsiderPairIntrinsic");

      DRange d_density_pos;
      d_density_pos.Init(0, 0);
      if (r_node.stat().count_pos > 0) {
	DRange dist_sq =
	  r_node.stat().bound_pos.RangeDistanceSq(q_node.bound());
	d_density_pos.hi = param.kernel_pos.EvalUnnormOnSq(dist_sq.lo);
	d_density_pos.lo = param.kernel_pos.EvalUnnormOnSq(dist_sq.hi);
      }

      DRange d_density_neg;
      d_density_neg.Init(0, 0);
      if (r_node.stat().count_neg > 0) {
	DRange dist_sq =
	  r_node.stat().bound_neg.RangeDistanceSq(q_node.bound());
	d_density_neg.hi = param.kernel_neg.EvalUnnormOnSq(dist_sq.lo);
	d_density_neg.lo = param.kernel_neg.EvalUnnormOnSq(dist_sq.hi);
      }

#if 1
      if ((d_density_pos.hi == 0 || d_density_pos.lo > 0)
	  && (d_density_neg.hi == 0 || d_density_neg.lo > 0)) {
	if (d_density_pos.lo > 0) {
	  q_postponed->moment_info_pos.Add(r_node.stat().moment_info_pos);
	}
	if (d_density_neg.lo > 0) {
	  q_postponed->moment_info_neg.Add(r_node.stat().moment_info_neg);
	}
        VERBOSE_MSG(1.0, "nbc: Intrinsic");
	return false;
      }
#else
      if ((d_density_pos.hi == 0)
	  && (d_density_neg.hi == 0)) {
        VERBOSE_MSG(1.0, "nbc: Intrinsic");
	return false;
      }
#endif

      d_density_pos *= r_node.stat().count_pos;
      d_density_neg *= r_node.stat().count_neg;
      delta->d_density_pos = d_density_pos;
      delta->d_density_neg = d_density_neg;

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

      // const_pos.lo <= const_pos_loo.lo, etc.; if !param.loo, ==
      // - tighter check possible given q's class, but effect is just tiny
      if (unlikely(param.coeff_pos
	   * q_summary_result.density_pos.lo * q_node.stat().pi_pos.lo
	   > q_summary_result.density_neg.hi * q_node.stat().pi_neg.hi)) {
	q_postponed->label = LAB_POS;
	return false;
      } else if (unlikely(param.coeff_neg
	   * q_summary_result.density_neg.lo * q_node.stat().pi_neg.lo
	   > q_summary_result.density_pos.hi * q_node.stat().pi_pos.hi)) {
	q_postponed->label = LAB_NEG;
	return false;
      }

      return true;
    }

    /**
     * Computes a heuristic for how early a computation should occur
     * -- smaller values are earlier.
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

  //fx_submodule(module, "io"); // influnce output order

  param.Init(fx_submodule(module, gnp_name));

  fx_timer_start(module, "read");
  param.no_labels = false;
  param.no_priors = fx_param_exists(module, "r_no_priors");
  r_points_cache = new DistributedCache();
  n_r_points = thor::ReadPoints<Nbc::RPoint>(
      param, DATA_CHANNEL + 0, DATA_CHANNEL + 1,
      fx_submodule(module, "r"),
      r_points_cache);
  if (fx_param_exists(module, "q")) {
    param.no_labels = fx_param_exists(module, "q_no_labels");
    param.no_priors = fx_param_exists(module, "q_no_priors");
    q_points_cache = new DistributedCache();
    n_q_points = thor::ReadPoints<Nbc::QPoint>(
        param, DATA_CHANNEL + 2, DATA_CHANNEL + 3,
	fx_submodule(module, "q"),
	q_points_cache);
  } else {
    q_points_cache = r_points_cache;
    n_q_points = n_r_points;
    param.loo = true;
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
      fx_submodule(module, "r_tree"),
      n_r_points, r_points_cache, r_tree);
  if (fx_param_exists(module, "q")) {
    q_tree = new ThorTree<Nbc::Param, Nbc::QPoint, Nbc::QNode>();
    thor::CreateKdTree<Nbc::QPoint, Nbc::QNode>(
        param, DATA_CHANNEL + 6, DATA_CHANNEL + 7,
	fx_submodule(module, "q_tree"),
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

  Nbc::GlobalResult global_result;
#if 0
  thor::RpcDualTree<Nbc, DualTreeDepthFirst<Nbc> >(
#else
  thor::RpcDualTree<Nbc, DualTreeRecursiveBreadth<Nbc> >(
#endif
      fx_submodule(module, "gnp"), GNP_CHANNEL, param,
      q_tree, r_tree, &q_results, &global_result);

  // Emit the results; this needs to be folded into THOR
  if (!fx_param_exists(module, "no_emit")) {
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
	classifications.set(0, (*points_iter).index(),
			    2 - (*result_iter).label);
      }
    }
    data::Save(fx_param_str(module, "out", "out.csv"), classifications);
  }

  rpc::Done();
}

int main(int argc, char *argv[]) {
  fx_module *root = fx_init(argc, argv, NULL);

  NbcMain(root);
  
  if (!rpc::is_root()) {
    fx_param_bool(root, "fx/silent", 1);
  }

  fx_done(root);
}
