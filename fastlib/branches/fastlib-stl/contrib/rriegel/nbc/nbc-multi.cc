#include "fastlib/fastlib_int.h"
#include "fastlib/thor/thor.h"

const fx_entry_doc nbc_multi_entries[] = {
  {"r", FX_REQUIRED, FX_STR, NULL,
   "  A reference data matrix file.  The last two columns contain\n"
   "  priors and labels, respectively, unless disabled.  Labels are\n"
   "  given 0 for negative and 1 for positive.\n"},
  {"r/no_priors", FX_PARAM, FX_BOOL, NULL,
   "  Indicates that r has no column for priors.\n"},
  {"q", FX_PARAM, FX_STR, NULL,
   "  A query data matrix file (default r).  As for r, the last two\n"
   "  columns contain priors and labels unless disabled.\n"},
  {"q/no_priors", FX_PARAM, FX_BOOL, NULL,
   "  Indicates that q has no column for priors.  If this is set, you\n"
   "  must also set --nbc/prior for proper computation.\n"},
  {"q/no_labels", FX_PARAM, FX_BOOL, NULL,
   "  Indicates that q has no column for labels.\n"},
  {"o", FX_PARAM, FX_STR, NULL,
   "  Destination file for classification with best bandwidths.\n"},
  {"o/no_emit", FX_PARAM, FX_BOOL, NULL,
   "  Disable the emition of results to file.\n"},
  {"nbc/max_h_pos", FX_REQUIRED, FX_DOUBLE, NULL,
   "  The greatest kernel bandwidth to test for the positive class.\n"},
  {"nbc/min_h_pos", FX_PARAM, FX_DOUBLE, NULL,
   "  The smallest bandwidth to test for the positive class (default 0);\n"
   "  min_h_pos + (max_h_pos - min_h_pos) / num_h_pos is the actual\n"
   "  smallest bandwidth tested.\n"},
  {"nbc/num_h_pos", FX_PARAM, FX_INT, NULL,
   "  Number of bandwidths for the positive class (default 1).\n"},
  {"nbc/max_h_neg", FX_REQUIRED, FX_DOUBLE, NULL,
   "  Like max_h_pos, but for negative class.\n"},
  {"nbc/min_h_neg", FX_PARAM, FX_DOUBLE, NULL,
   "  Like min_h_pos, but for negative class.\n"},
  {"nbc/num_h_neg", FX_PARAM, FX_INT, NULL,
   "  Like num_h_pos, but for negative class.\n"},
  {"nbc/prior", FX_PARAM, FX_DOUBLE, NULL,
   "  Overrides the columns of priors for q.  Set this if --q/no_priors\n"
   "  is given.\n"},
  {"nbc/threshold", FX_PARAM, FX_DOUBLE, NULL,
   "  Positive class postierior probability required to classify as\n"
   "  positive; use 0.5 for Bayes optimal classifier (default 0.5).\n"},
  {"read", FX_TIMER, FX_CUSTOM, NULL,
   "  Time spent reading data from file.\n"},
  {"tree", FX_TIMER, FX_CUSTOM, NULL,
   "  Time spent building trees.\n"},
  {"gnp/gnp", FX_TIMER, FX_CUSTOM, NULL,
   "  Time spent in generaled N-body computation.\n"},
  {"gnp/global_result/best_eff_pos", FX_RESULT, FX_DOUBLE, NULL,
   "  Best observed efficiency for postive class (percentage classified\n"
   "  positive points actually positive).\n"},
  {"gnp/global_result/best_cov_pos", FX_RESULT, FX_DOUBLE, NULL,
   "  Best observed coverage for postive class (percentage actually\n"
   "  positive points classified positive).\n"},
  {"gnp/global_result/best_eff_neg", FX_RESULT, FX_DOUBLE, NULL,
   "  Like best_eff_pos, but for negative class.\n"},
  {"gnp/global_result/best_cov_neg", FX_RESULT, FX_DOUBLE, NULL,
   "  Like best_cov_pos, but for negative class.\n"},
  {"gnp/global_result/best_h_pos", FX_RESULT, FX_DOUBLE, NULL,
   "  Best observed bandwidth for the positive class.\n"},
  {"gnp/global_result/best_h_neg", FX_RESULT, FX_DOUBLE, NULL,
   "  Best observed bandwidth for the negative class.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc nbc_multi_doc = {
  nbc_multi_entries, NULL,
  "This program performes nonparametric Bayes classification with multiple\n"
  "bandwidths for the purpose of bandwidth optimization.\n"
};

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
      mem::BitCopy(vec_.ptr(), data.ptr(), vec_.length());
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

  struct Coeff {
   public:
    double std;
    double loo;

    OT_DEF_BASIC(Coeff) {
      OT_MY_OBJECT(std);
      OT_MY_OBJECT(loo);
    }
  };

  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by THOR.
   */
  struct Param {
   public:
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
    /** Number of positive kernels to test. Similar for _neg. */
    index_t count_kernel_pos;
    index_t count_kernel_neg;
    /** The kernels for positive points. Similar for _neg. */
    ArrayList<Kernel> kernel_pos;
    ArrayList<Kernel> kernel_neg;
    /**
     * Normalization coeff for positive points. Similar for _neg.
     *
     * .loo versions should be used to classify a point as its own
     * class in the LOO case.  Note that the .std coeff is necessarily
     * smaller than .loo, and thus suitable for all lower bounds.
     * Example use:
     *
     *   coeff_pos.std * density_pos.lo * pi_pos.lo
     *     > coeff_neg.loo * density_neg.hi * pi_neg.hi
     *
     * is tight for classifying a negative point as positive and won't
     * misclassify positive points.
     */
    ArrayList<Coeff> coeff_pos;
    ArrayList<Coeff> coeff_neg;
    /** Whether to compute densities leaving out matching indices. */
    bool loo;
    /** Whether to read labels from current set. Similar for _priors. */
    bool no_labels;
    bool no_priors;

    OT_DEF(Param) {
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(count_all);
      OT_MY_OBJECT(count_pos);
      OT_MY_OBJECT(count_neg);
      OT_MY_OBJECT(count_kernel_pos);
      OT_MY_OBJECT(count_kernel_neg);
      OT_MY_OBJECT(kernel_pos);
      OT_MY_OBJECT(kernel_neg);
      OT_MY_OBJECT(prior_override);
      OT_MY_OBJECT(coeff_pos);
      OT_MY_OBJECT(coeff_neg);
      OT_MY_OBJECT(loo);
      OT_MY_OBJECT(no_labels);
      OT_MY_OBJECT(no_priors);
    }

   public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      DRange h_pos;
      DRange h_neg;
      double step;
      double h_cur;
      int i;

      count_kernel_pos = fx_param_int(module, "num_h_pos", 1);
      kernel_pos.Init(count_kernel_pos);
      coeff_pos.Init(count_kernel_pos);

      h_pos.hi = fx_param_double_req(module, "max_h_pos");
      h_pos.lo = fx_param_double(module, "min_h_pos", 0);

      step = h_pos.width() / count_kernel_pos;
      h_cur = h_pos.hi;
      for (i = count_kernel_pos - 1; i >= 0; --i, h_cur -= step) {
	kernel_pos[i].Init(h_cur);
      }

      count_kernel_neg = fx_param_int(module, "num_h_neg", 1);
      kernel_neg.Init(count_kernel_neg);
      coeff_neg.Init(count_kernel_neg);

      h_neg.hi = fx_param_double_req(module, "max_h_neg");
      h_neg.lo = fx_param_double(module, "min_h_neg", 0);

      step = h_neg.width() / count_kernel_neg;
      h_cur = h_neg.hi;
      for (i = count_kernel_neg - 1; i >= 0; --i, h_cur -= step) {
	kernel_neg[i].Init(h_cur);
      }

      threshold = fx_param_double(module, "threshold", 0.5);
      prior_override = fx_param_double(module, "prior", -1.0);

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
    }

    /**
     * Finalize parameters (Not THOR).
     */
    void ComputeConsts(int count_pos_in, int count_neg_in) {
      int i;

      count_pos = count_pos_in;
      count_neg = count_neg_in;

      index_t count_pos_loo = loo ? count_pos - 1 : count_pos;
      index_t count_neg_loo = loo ? count_neg - 1 : count_neg;

      for (i = 0; i < count_kernel_pos; i++) {
	double norm_pos = kernel_pos[i].CalcNormConstant(dim);

	coeff_pos[i].std = (1 - threshold) / (norm_pos * count_pos);
	coeff_pos[i].loo = (1 - threshold) / (norm_pos * count_pos_loo);
      }

      for (i = 0; i < count_kernel_neg; i++) {
	double norm_neg = kernel_neg[i].CalcNormConstant(dim);

	coeff_neg[i].std = threshold / (norm_neg * count_neg);
	coeff_neg[i].loo = threshold / (norm_neg * count_neg_loo);
      }

      ot::Print(*this);
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

  struct MultiLabel {
   public:
    /** Size of the label matrix. Similar for j_. */
    index_t i_size;
    index_t j_size;
    /** The labels, stored column-major. */
    ArrayList<int> label;
    /** Ranges for unlabeled portion. Similar for _hi and j_.
     *
     * Invariant: for any i < i_lo, i >= i_hi, j < j_lo, or j >= j_hi,
     *
     *   label.at(i,j) != LAB_EITHER
     *
     * Thus, no further work is necessary for kernels outside the
     * lo-hi range.
     */
    index_t i_lo;
    index_t i_hi;
    index_t j_lo;
    index_t j_hi;

    OT_DEF(MultiLabel) {
      OT_MY_OBJECT(i_size);
      OT_MY_OBJECT(j_size);
      OT_MY_OBJECT(label);
      OT_MY_OBJECT(i_lo);
      OT_MY_OBJECT(i_hi);
      OT_MY_OBJECT(j_lo);
      OT_MY_OBJECT(j_hi);
    }

   public:
    void Init(index_t i_size_in, index_t j_size_in) {
      i_size = i_size_in;
      j_size = j_size_in;
      label.Init(i_size * j_size);

      Reset();
    }

    void Reset() {
      i_lo = 0;
      i_hi = i_size;
      j_lo = 0;
      j_hi = j_size;

      SetAll(LAB_EITHER);
    }

    void SetAll(Label label_in) {
      for (int i = 0; i < label.size(); i++) {
	label[i] = label_in;
      }
    }

    void SetEqual(const MultiLabel& other) {
      DEBUG_SAME_SIZE(label.size(), other.label.size());

      for (int i = 0; i < label.size(); i++) {
	label[i] = other.label[i];
      }

      i_lo = other.i_lo;
      i_hi = other.i_hi;
      j_lo = other.j_lo;
      j_hi = other.j_hi;
    }

    int& at(index_t i, index_t j) {
      return label[i + j * i_size];
    }

    const int& at(index_t i, index_t j) const {
      return label[i + j * i_size];
    }

    MultiLabel& operator&= (const MultiLabel& other) {
      DEBUG_SAME_SIZE(label.size(), other.label.size());

      index_t next_i_lo = i_size;
      index_t next_i_hi = 0;
      index_t next_j_lo = j_size;
      index_t next_j_hi = 0;

      for (index_t j = 0; j < j_size; j++) {
	for (index_t i = 0; i < i_size; i++) {
	  at(i,j) &= other.at(i,j);
	  DEBUG_ASSERT_MSG(at(i,j) != LAB_NEITHER, "Conflicting labels?");

	  if (at(i,j) == LAB_EITHER) {
	    next_i_lo = std::min(next_i_lo, i);
	    next_i_hi = std::max(next_i_hi, i+1);
	    next_j_lo = std::min(next_j_lo, j);
	    next_j_hi = std::max(next_j_hi, j+1);
	  }
	}
      }

      i_lo = next_i_lo;
      i_hi = next_i_hi;
      j_lo = next_j_lo;
      j_hi = next_j_hi;

      return *this;
    }

    MultiLabel& operator|= (const MultiLabel& other) {
      DEBUG_SAME_SIZE(label.size(), other.label.size());

      index_t next_i_lo = i_size;
      index_t next_i_hi = 0;
      index_t next_j_lo = j_size;
      index_t next_j_hi = 0;

      for (index_t j = 0; j < j_size; j++) {
	for (index_t i = 0; i < i_size; i++) {
	  at(i,j) |= other.at(i,j);

	  if (at(i,j) == LAB_EITHER) {
	    next_i_lo = std::min(next_i_lo, i);
	    next_i_hi = std::max(next_i_hi, i+1);
	    next_j_lo = std::min(next_j_lo, j);
	    next_j_hi = std::max(next_j_hi, j+1);
	  }
	}
      }

      i_lo = next_i_lo;
      i_hi = next_i_hi;
      j_lo = next_j_lo;
      j_hi = next_j_hi;

      return *this;
    }
  };

  /**
   * Coarse result on a region.
   */
  struct QPostponed {
   public:
    /** Moments of pruned things. */
    ArrayList<MomentInfo> moment_info_pos;
    ArrayList<MomentInfo> moment_info_neg;
    /** We pruned an entire part of the tree with a particular label. */
    MultiLabel label;

    OT_DEF(QPostponed) {
      OT_MY_OBJECT(moment_info_pos);
      OT_MY_OBJECT(moment_info_neg);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      label.Init(param.count_kernel_pos, param.count_kernel_neg);

      moment_info_pos.Init(param.count_kernel_pos);
      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	moment_info_pos[i].Init(param);
      }
      moment_info_neg.Init(param.count_kernel_neg);
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	moment_info_neg[j].Init(param);
      }

    }

    void Reset(const Param& param) {
      label.Reset();

      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	moment_info_pos[i].Reset();
      }
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	moment_info_neg[j].Reset();
      }
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      // Collect labels and tighten index ranges
      label &= other.label;

      // Update moments for all indices; perhaps should be unlabeled
      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	moment_info_pos[i].Add(other.moment_info_pos[i]);
      }
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	moment_info_neg[j].Add(other.moment_info_neg[j]);
      }
    }
  };

  /**
   * Coarse result on a region.
   */
  struct Delta {
   public:
    /** Density update to apply to children's bound. Similar for _neg. */
    ArrayList<DRange> d_density_pos;
    ArrayList<DRange> d_density_neg;
    /** Ranges for relevant portion. Similar for _hi and j_.
     *
     * Kernels outside of this range have already been pruned for this
     * query-reference combination (intrinsically or terminally,
     * though terminal prunes are represented elsewhere, too).
     */
    index_t i_lo;
    index_t i_hi;
    index_t j_lo;
    index_t j_hi;

    OT_DEF(Delta) {
      OT_MY_OBJECT(d_density_pos);
      OT_MY_OBJECT(d_density_neg);
      OT_MY_OBJECT(i_lo);
      OT_MY_OBJECT(i_hi);
      OT_MY_OBJECT(j_lo);
      OT_MY_OBJECT(j_hi);
    }

   public:
    /**
     * Prepares a delta for use.  Reg. THOR.
     *
     * Note: must be suitable for use as parent delta of root.
     */
    void Init(const Param& param) {
      d_density_pos.Init(param.count_kernel_pos);
      d_density_neg.Init(param.count_kernel_neg);

      i_lo = 0;
      i_hi = param.count_kernel_pos;
      j_lo = 0;
      j_hi = param.count_kernel_neg;
    }
  };

  struct QResult {
   public:
    /** Analytically computed densities. Similar for _neg. */
    ArrayList<double> density_pos;
    ArrayList<double> density_neg;
    /** Labels for the point under the various kernel bandwidths. */
    MultiLabel label;

    OT_DEF(QResult) {
      OT_MY_OBJECT(density_pos);
      OT_MY_OBJECT(density_neg);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      label.Init(param.count_kernel_pos, param.count_kernel_neg);

      density_pos.Init(param.count_kernel_pos);
      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	density_pos[i] = 0;
      }
      density_neg.Init(param.count_kernel_neg);
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	density_neg[j] = 0;
      }
    }

    void Seed(const Param& param, const QPoint& q) {
      if (param.loo) {
	if (q.is_pos()) {
	  for (index_t i = 0; i < param.count_kernel_pos; i++) {
	    density_pos[i] -= param.kernel_pos[i].EvalUnnormOnSq(0);
	  }
	} else {
	  for (index_t j = 0; j < param.count_kernel_neg; j++) {
	    density_neg[j] -= param.kernel_neg[j].EvalUnnormOnSq(0);
	  }
	}
      }
    }

    // Why does this have a q_index and r_root?  RR
    void Postprocess(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_root) {
      for (index_t j = label.j_lo; j < label.j_hi; j++) {
	for (index_t i = label.i_lo; i < label.i_hi; i++) {
	  if (label.at(i,j) == LAB_EITHER) {
	    // Use proper coeffs for q; branches are equiv if not LOO
	    if (q.is_pos()) {
	      if (param.coeff_pos[i].loo * density_pos[i] * q.pi_pos()
		  > param.coeff_neg[j].std * density_neg[j] * q.pi_neg()) {
		label.at(i,j) &= LAB_POS;
	      } else if (param.coeff_neg[j].std * density_neg[j] * q.pi_neg()
		  > param.coeff_pos[i].loo * density_pos[i] * q.pi_pos()) {
		label.at(i,j) &= LAB_NEG;
	      }
	    } else {
	      if (param.coeff_pos[i].std * density_pos[i] * q.pi_pos()
		  > param.coeff_neg[j].loo * density_neg[j] * q.pi_neg()) {
		label.at(i,j) &= LAB_POS;
	      } else if (param.coeff_neg[j].loo * density_neg[j] * q.pi_neg()
		  > param.coeff_pos[i].std * density_pos[i] * q.pi_pos()) {
		label.at(i,j) &= LAB_NEG;
	      }
	    }
	  }
	}
      }
    }

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const QPoint& q, index_t q_index) {
      label &= postponed.label;

      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	if (!postponed.moment_info_pos[i].is_empty()) {
	  density_pos[i] += postponed.moment_info_pos[i].ComputeKernelSum(
              param.kernel_pos[i], q.vec());
	}
      }
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	if (!postponed.moment_info_neg[j].is_empty()) {
	  density_neg[j] += postponed.moment_info_neg[j].ComputeKernelSum(
              param.kernel_neg[j], q.vec());
	}
      }
    }
  };

  struct QSummaryResult {
   public:
    /** Bound on density from leaves. Similar for _neg. */
    ArrayList<DRange> density_pos;
    ArrayList<DRange> density_neg;
    /** Labels that apply to the entire node. */
    MultiLabel label;

    OT_DEF(QSummaryResult) {
      OT_MY_OBJECT(density_pos);
      OT_MY_OBJECT(density_neg);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      /* horizontal init */
      label.Init(param.count_kernel_pos, param.count_kernel_neg);

      density_pos.Init(param.count_kernel_pos);
      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	density_pos[i].Init(0, 0);
      }
      density_neg.Init(param.count_kernel_neg);
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	density_neg[j].Init(0, 0);
      }
    }

    void Seed(const Param& param, const QNode& q_node) {
      if (param.loo) {
	if (q_node.stat().count_pos > 0) {
	  for (index_t i = label.i_lo; i < label.i_hi; i++) {
	    density_pos[i].lo -= param.kernel_pos[i].EvalUnnormOnSq(0);
	  }
	} else {
	  for (index_t j = label.j_lo; j < label.j_hi; j++) {
	    density_neg[j].hi -= param.kernel_neg[j].EvalUnnormOnSq(0);
	  }
	}

	if (q_node.stat().count_neg > 0) {
	  for (index_t j = label.j_lo; j < label.j_hi; j++) {
	    density_neg[j].lo -= param.kernel_neg[j].EvalUnnormOnSq(0);
	  }
	} else {
	  for (index_t i = label.i_lo; i < label.i_hi; i++) {
	    density_pos[i].hi -= param.kernel_pos[i].EvalUnnormOnSq(0);
	  }
	}
      }
    }

    // Why does this have a q_node?  RR
    void StartReaccumulate(const Param& param, const QNode& q_node) {
      /* vertical init */
      label.SetAll(LAB_NEITHER);

      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	density_pos[i].InitEmptySet();
      }
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	density_neg[j].InitEmptySet();
      }
    }

    void Accumulate(const Param& param, const QResult& result) {
      label |= result.label;

      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	density_pos[i] |= result.density_pos[i];
      }
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	density_neg[j] |= result.density_neg[j];
      }
    }

    void Accumulate(const Param& param,
        const QSummaryResult& result, index_t n_points) {
      label |= result.label;

      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	density_pos[i] |= result.density_pos[i];
      }
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	density_neg[j] |= result.density_neg[j];
      }
    }

    void FinishReaccumulate(const Param& param,
        const QNode& q_node) {
      /* no post-processing steps necessary */
    }

    /** horizontal join operator */
    void ApplySummaryResult(const Param& param,
        const QSummaryResult& summary_result) {
      label &= summary_result.label;

      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	density_pos[i] += summary_result.density_pos[i];
      }
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	density_neg[j] += summary_result.density_neg[j];
      }
    }

    void ApplyDelta(const Param& param,
        const Delta& delta) {
      for (index_t i = delta.i_lo; i < delta.i_hi; i++) {
	density_pos[i] += delta.d_density_pos[i];
      }
      for (index_t j = delta.j_lo; j < delta.j_hi; j++) {
	density_neg[j] += delta.d_density_neg[j];
      }
    }

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      label &= postponed.label;

      for (index_t i = 0; i < param.count_kernel_pos; i++) {
	if (!postponed.moment_info_pos[i].is_empty()) {
	  density_pos[i] += postponed.moment_info_pos[i].ComputeKernelSumRange(
              param.kernel_pos[i], q_node.bound());
	}
      }
      for (index_t j = 0; j < param.count_kernel_neg; j++) {
	if (!postponed.moment_info_neg[j].is_empty()) {
	  density_neg[j] += postponed.moment_info_neg[j].ComputeKernelSumRange(
              param.kernel_neg[j], q_node.bound());
	}
      }

    }
  };

  /**
   * A simple postprocess-step global result.
   */
  struct GlobalResult {
   public:
    /** Size of the results matrix. Similar for j_. */
    index_t i_size;
    index_t j_size;
    /** Identified pos points, stored column-major. Similar for _neg. */
    ArrayList<index_t> count_pos;
    ArrayList<index_t> count_neg;
    /** Number of points that couldn't be classified. */
    ArrayList<index_t> count_unknown;
    /** Correct pos points, stored column-major. Similar for _neg. */
    ArrayList<index_t> count_correct_pos;
    ArrayList<index_t> count_correct_neg;
    index_t count_true_pos;
    index_t count_true_neg;

    OT_DEF(GlobalResult) {
      OT_MY_OBJECT(i_size);
      OT_MY_OBJECT(j_size);
      OT_MY_OBJECT(count_pos);
      OT_MY_OBJECT(count_neg);
      OT_MY_OBJECT(count_unknown);
      OT_MY_OBJECT(count_correct_pos);
      OT_MY_OBJECT(count_correct_neg);
      OT_MY_OBJECT(count_true_pos);
      OT_MY_OBJECT(count_true_neg);
    }

   public:
    void Init(const Param& param) {
      i_size = param.kernel_pos.size();
      j_size = param.kernel_neg.size();
      count_pos.Init(i_size * j_size);
      count_neg.Init(i_size * j_size);
      count_unknown.Init(i_size * j_size);
      count_correct_pos.Init(i_size * j_size);
      count_correct_neg.Init(i_size * j_size);

      for (int i = 0; i < count_pos.size(); i++) {
	count_pos[i] = 0;
	count_neg[i] = 0;
	count_unknown[i] = 0;
	count_correct_pos[i] = 0;
	count_correct_neg[i] = 0;
      }

      count_true_pos = 0;
      count_true_neg = 0;
    }

    void Accumulate(const Param& param, const GlobalResult& other) {
      for (int i = 0; i < count_pos.size(); i++) {
	count_pos[i] += other.count_pos[i];
	count_neg[i] += other.count_neg[i];
	count_unknown[i] += other.count_unknown[i];
	count_correct_pos[i] += other.count_correct_pos[i];
	count_correct_neg[i] += other.count_correct_neg[i];
      }

      count_true_pos += other.count_true_pos;
      count_true_neg += other.count_true_neg;
    }

    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}
    void Postprocess(const Param& param) {}

    void Report(const Param& param, datanode *datanode) {
      if (likely(param.loo)) {
	double best_eff_pos = 0;
	double best_eff_neg = 0;
	double best_cov_pos = 0;
	double best_cov_neg = 0;
	index_t best_i = -1;
	index_t best_j = -1;

	for (index_t j = 0; j < j_size; j++) {
	  for (index_t i = 0; i < i_size; i++) {
	    double eff_pos =
	        count_correct_pos[i+j*i_size] / (double)count_pos[i+j*i_size];
	    double eff_neg =
	        count_correct_neg[i+j*i_size] / (double)count_neg[i+j*i_size];

	    double cov_pos =
	        count_correct_pos[i+j*i_size] / (double)count_true_pos;
	    double cov_neg =
	        count_correct_neg[i+j*i_size] / (double)count_true_neg;

	    if (eff_pos * cov_pos > best_eff_pos * best_cov_pos) {
	      best_eff_pos = eff_pos;
	      best_cov_pos = cov_pos;
	      best_eff_neg = eff_neg;
	      best_cov_neg = cov_neg;
	      best_i = i;
	      best_j = j;
	    }
	  }
	}

	if (best_i != -1 && best_j != -1) {
	  fx_result_double(datanode, "best_eff_pos", best_eff_pos);
	  fx_result_double(datanode, "best_cov_pos", best_cov_pos);
	  fx_result_double(datanode, "best_eff_neg", best_eff_neg);
	  fx_result_double(datanode, "best_cov_neg", best_cov_neg);
	  fx_result_double(datanode, "best_h_pos",
	      sqrt(param.kernel_pos[best_i].bandwidth_sq()));
	  fx_result_double(datanode, "best_h_neg",
	      sqrt(param.kernel_neg[best_j].bandwidth_sq()));
	} else {
	  fx_result_double(datanode, "best_eff_pos", 0);
	  fx_result_double(datanode, "best_eff_neg", 0);
	  fx_result_double(datanode, "best_cov_pos", 0);
	  fx_result_double(datanode, "best_cov_neg", 0);
	  fx_result_double(datanode, "best_h_pos", -1);
	  fx_result_double(datanode, "best_h_neg", -1);
	}
      }
    }

    void ApplyResult(const Param& param,
        const QPoint& q_point, index_t q_i,
        const QResult& q_result) {
      for (int i = 0; i < count_pos.size(); ++i) {
	if (q_result.label.label[i] == LAB_POS) {
	  ++count_pos[i];
	  if (!param.no_labels && q_point.is_pos()) {
	    ++count_correct_pos[i];
	  }
	} else if (q_result.label.label[i] == LAB_NEG) {
	  ++count_neg[i];
	  if (!param.no_labels && !q_point.is_pos()) {
	    ++count_correct_neg[i];
	  }
	} else if (q_result.label.label[i] == LAB_EITHER) {
	  ++count_unknown[i];
	}
      }
      if (!param.no_labels) {
	if (q_point.is_pos()) {
	  ++count_true_pos;
	} else {
	  ++count_true_neg;
	}
      }
    }
  };

  /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  struct PairVisitor {
   public:
    ArrayList<double> d_density_pos;
    ArrayList<double> d_density_neg;
    index_t i_lo;
    index_t i_hi;
    index_t j_lo;
    index_t j_hi;

   public:
    void Init(const Param& param) {
      d_density_pos.Init(param.count_kernel_pos);
      d_density_neg.Init(param.count_kernel_neg);
    }

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
      // *Should* be sufficient to just check one of i or j
      if (unlikely(q_result->label.i_lo >= q_result->label.i_hi
	   || q_result->label.j_lo >= q_result->label.j_hi)) {
        return false;
      }

      // Restrict computation to things not already pruned for this pair

      if (r_node.stat().count_pos > 0) {
	DRange dist_sq = r_node.stat().bound_pos.RangeDistanceSq(q.vec());

	for (i_lo = delta.i_lo; i_lo < delta.i_hi; i_lo++) {
	  if (likely(dist_sq.lo < param.kernel_pos[i_lo].bandwidth_sq())) {
	    break;
	  }
	}
	for (i_hi = delta.i_hi; i_hi > i_lo; i_hi--) {
	  if (likely(dist_sq.hi > param.kernel_pos[i_hi-1].bandwidth_sq())) {
	    break;
	  } else {
	    q_result->density_pos[i_hi-1] +=
	      r_node.stat().moment_info_pos.ComputeKernelSum(
		  param.kernel_pos[i_hi-1], q.vec());
	  }
	}
      } else {
	i_lo = 0;
	i_hi = 0;
      }

      if (r_node.stat().count_neg > 0) {
	DRange dist_sq = r_node.stat().bound_neg.RangeDistanceSq(q.vec());

	for (j_lo = delta.j_lo; j_lo < delta.j_hi; j_lo++) {
	  if (likely(dist_sq.lo < param.kernel_neg[j_lo].bandwidth_sq())) {
	    break;
	  }
	}
	for (j_hi = delta.j_hi; j_hi > j_lo; j_hi--) {
	  if (likely(dist_sq.hi > param.kernel_neg[j_hi-1].bandwidth_sq())) {
	    break;
	  } else {
	    q_result->density_neg[j_hi-1] +=
	      r_node.stat().moment_info_neg.ComputeKernelSum(
		  param.kernel_neg[j_hi-1], q.vec());
	  }
	}
      } else {
	j_lo = 0;
	j_hi = 0;
      }

      if (unlikely(i_lo >= i_hi && j_lo >= j_hi)) {
	return false;
      }

      for (index_t i = i_lo; i < i_hi; i++) {
	d_density_pos[i] = 0;
      }

      for (index_t j = j_lo; j < j_hi; j++) {
	d_density_neg[j] = 0;
      }

      return true;
    }

    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
      double distance = la::DistanceSqEuclidean(q.vec(), r.vec());
      if (r.is_pos()) {
	for (index_t i = i_lo; i < i_hi; i++) {
	  d_density_pos[i] += param.kernel_pos[i].EvalUnnormOnSq(distance);
	}
      } else {
	for (index_t j = j_lo; j < j_hi; j++) {
	  d_density_neg[j] += param.kernel_neg[j].EvalUnnormOnSq(distance);
	}
      }
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_node,
        const QSummaryResult& unapplied,
        QResult* q_result,
        GlobalResult* global_result) {
      index_t i;
      index_t j;

      for (i = i_lo; i < i_hi; i++) {
	q_result->density_pos[i] += d_density_pos[i];
      }
      for (j = j_lo; j < j_hi; j++) {
	q_result->density_neg[j] += d_density_neg[j];
      }

      index_t next_i_lo = q_result->label.i_hi - 1;
      index_t next_i_hi = q_result->label.i_lo;
      index_t next_j_lo = q_result->label.j_hi - 1;
      index_t next_j_hi = q_result->label.j_lo;

      // Intentionally wider ranges than above
      // - to perform classifications for pervious intrinsic prunes
      for (j = q_result->label.j_lo; j < q_result->label.j_hi; j++) {
	for (i = q_result->label.i_lo; i < q_result->label.i_hi; i++) {
	  if (q_result->label.at(i,j) == LAB_EITHER) {
	    DRange total_density_pos =
	      unapplied.density_pos[i] + q_result->density_pos[i];
	    DRange total_density_neg =
	      unapplied.density_neg[j] + q_result->density_neg[j];

	    // const_pos.lo <= const_pos_loo.lo, etc.; if !param.loo, ==
	    // - tighter check possible given q's class, but effect is tiny
	    if (unlikely(
		 param.coeff_pos[i].std * total_density_pos.lo * q.pi_pos() >
		 param.coeff_neg[j].loo * total_density_neg.hi * q.pi_neg())) {
	      q_result->label.at(i,j) &= LAB_POS;
	    } else if (unlikely(
		 param.coeff_neg[j].std * total_density_neg.lo * q.pi_neg() >
		 param.coeff_pos[i].loo * total_density_pos.hi * q.pi_pos())) {
	      q_result->label.at(i,j) &= LAB_NEG;
	    } else {
	      next_i_lo = std::min(next_i_lo, i);
	      next_i_hi = std::max(next_i_hi, i+1);
	      next_j_lo = std::min(next_j_lo, j);
	      next_j_hi = std::max(next_j_hi, j+1);
	    }
	  }
	}
      }

      q_result->label.i_lo = next_i_lo;
      q_result->label.i_hi = next_i_hi;
      q_result->label.j_lo = next_j_lo;
      q_result->label.j_hi = next_j_hi;
    }
  };

  class Algorithm {
   public:
    /**
     * Calculates a delta....
     *
     * I'm hoping these can be lies:
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
      index_t i_lo;
      index_t i_hi;
      index_t j_lo;
      index_t j_hi;

      // Restrict computation to things not already pruned for this pair

      if (r_node.stat().count_pos > 0) {
	DRange dist_sq =
	  r_node.stat().bound_pos.RangeDistanceSq(q_node.bound());

	for (i_hi = parent_delta.i_hi; i_hi > parent_delta.i_lo; i_hi--) {
	  if (likely(dist_sq.hi > param.kernel_pos[i_hi-1].bandwidth_sq())) {
	    break;
	  } else {
	    q_postponed->moment_info_pos[i_hi-1].Add(
		r_node.stat().moment_info_pos);
	  }
	}
	for (i_lo = i_hi; i_lo > parent_delta.i_lo; i_lo--) {
	  if (likely(dist_sq.lo < param.kernel_pos[i_lo-1].bandwidth_sq())) {
	    delta->d_density_pos[i_lo-1].lo = 0;
	    delta->d_density_pos[i_lo-1].hi = r_node.stat().count_pos
	      * param.kernel_pos[i_lo-1].EvalUnnormOnSq(dist_sq.lo);
	  } else {
	    break;
	  }
	}
      } else {
	i_lo = 0;
	i_hi = 0;
      }

      if (r_node.stat().count_neg > 0) {
	DRange dist_sq =
	  r_node.stat().bound_neg.RangeDistanceSq(q_node.bound());

	for (j_hi = parent_delta.j_hi; j_hi > parent_delta.j_lo; j_hi--) {
	  if (likely(dist_sq.hi > param.kernel_neg[j_hi-1].bandwidth_sq())) {
	    break;
	  } else {
	    q_postponed->moment_info_neg[j_hi-1].Add(
		r_node.stat().moment_info_neg);
	  }
	}
	for (j_lo = j_hi; j_lo > parent_delta.j_lo; j_lo--) {
	  if (likely(dist_sq.lo < param.kernel_neg[j_lo-1].bandwidth_sq())) {
	    delta->d_density_neg[j_lo-1].lo = 0;
	    delta->d_density_neg[j_lo-1].hi = r_node.stat().count_neg
	      * param.kernel_neg[j_lo-1].EvalUnnormOnSq(dist_sq.lo);
	  } else {
	    break;
	  }
	}
      } else {
	j_lo = 0;
	j_hi = 0;
      }

      if (unlikely(i_lo >= i_hi && j_lo >= j_hi)) {
	return false;
      }

      delta->i_lo = i_lo;
      delta->i_hi = i_hi;
      delta->j_lo = j_lo;
      delta->j_hi = j_hi;

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
      q_postponed->label.SetEqual(q_summary_result.label);

      if (unlikely(q_summary_result.label.i_lo >= q_summary_result.label.i_hi
	   || q_summary_result.label.j_lo >= q_summary_result.label.j_hi)) {
	return false;
      }

      index_t next_i_lo = q_summary_result.label.i_hi + 1;
      index_t next_i_hi = q_summary_result.label.i_lo;
      index_t next_j_lo = q_summary_result.label.j_hi + 1;
      index_t next_j_hi = q_summary_result.label.j_lo;

      for (index_t j = q_summary_result.label.j_lo;
	   j < q_summary_result.label.j_hi; j++) {
	for (index_t i = q_summary_result.label.i_lo;
	     i < q_summary_result.label.i_hi; i++) {
	  if (q_summary_result.label.at(i,j) == LAB_EITHER) {
	    // const_pos.lo <= const_pos_loo.lo, etc.; if !param.loo, ==
	    // - tighter check possible given q's class, but effect is tiny
	    if (unlikely(
		 param.coeff_pos[i].std
		 * q_summary_result.density_pos[i].lo
		 * q_node.stat().pi_pos.lo
		 > param.coeff_neg[j].loo
		 * q_summary_result.density_neg[j].hi
		 * q_node.stat().pi_neg.hi)) {
	      q_postponed->label.at(i,j) &= LAB_POS;
	    } else if (unlikely(
		 param.coeff_neg[j].std
		 * q_summary_result.density_neg[j].lo
		 * q_node.stat().pi_neg.lo
		 > param.coeff_pos[i].loo
		 * q_summary_result.density_pos[i].hi
		 * q_node.stat().pi_pos.hi)) {
	      q_postponed->label.at(i,j) &= LAB_NEG;
	    } else {
	      next_i_lo = std::min(next_i_lo, i);
	      next_i_hi = std::max(next_i_hi, i+1);
	      next_j_lo = std::min(next_j_lo, j);
	      next_j_hi = std::max(next_j_hi, j+1);
	    }
	  }
	}
      }

      q_postponed->label.i_lo = next_i_lo;
      q_postponed->label.i_hi = next_i_hi;
      q_postponed->label.j_lo = next_j_lo;
      q_postponed->label.j_hi = next_j_hi;

      if (unlikely(next_i_lo >= next_i_hi || next_j_lo >= next_j_hi)) {
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
  param.no_priors = fx_param_bool(module, "r/no_priors", 0);
  r_points_cache = new DistributedCache();
  n_r_points = thor::ReadPoints<Nbc::RPoint>(
      param, DATA_CHANNEL + 0, DATA_CHANNEL + 1,
      fx_submodule(module, "r"),
      r_points_cache);
  if (fx_param_exists(module, "q")) {
    param.no_labels = fx_param_bool(module, "q/no_labels", 0);
    param.no_priors = fx_param_bool(module, "q/no_priors", 0);
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
  const char *out_file = fx_param_str(module, "o", "out.csv");
  if (out_file[0] != '\0' && !fx_param_bool(module, "o/no_emit", 0)) {
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
			    2 - (*result_iter).label.at(0,0));
      }
    }
    data::Save(out_file, classifications);
  }

  rpc::Done();
}

int main(int argc, char *argv[]) {
  fx_module *root = fx_init(argc, argv, &nbc_multi_doc);

  NbcMain(root);
  
  if (!rpc::is_root()) {
    fx_param_bool(root, "fx/silent", 1);
  }

  fx_done(root);
}
