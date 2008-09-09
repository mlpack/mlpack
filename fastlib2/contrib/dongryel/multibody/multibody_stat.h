#ifndef INSIDE_MULTIBODY_H
#error "This is not a public header file!"
#endif

#ifndef MULTIBODY_STAT_H
#define MULTIBODY_STAT_H

class MultibodyStat {

 public:
  
  ////////// Member Variables //////////

  /** @brief The total coordinate sum.
   */
  Vector coordinate_sum_;

  /** @brief The L1 norm of the total coordinate sum.
   */
  double l1_norm_coordinate_sum_;

  /** @brief The leave-one-out lower bound on the L1 norm of the total
   *         coordinate sum.
   */
  double l1_norm_coordinate_sum_l_;

  /** @brief The maximum L1 norm of the points owned by this
   *         statistics.
   */
  double max_l1_norm_;

  /** @brief The maximum negative gradient (first component).
   */
  double negative_gradient1_u;

  /** @brief The used error for the first negative gradient component.
   */
  double negative_gradient1_used_error;

  /** @brief The minimum positive gradient (first component).
   */
  double positive_gradient1_l;

  /** @brief The used error for the first positive gradient component.
   */
  double positive_gradient1_used_error;
  
  /** @brief The L1 norm of maximum negative gradient (second component).
   */
  double l1_norm_negative_gradient2_u;

  /** @brief The used error for the second negative gradient
   *         component.
   */
  double negative_gradient2_used_error;
  
  /** @brief The L1 norm of minimum positive gradient (second component).
   */
  double l1_norm_positive_gradient2_l;

  /** @brief The used error for the second positive gradient component.
   */
  double positive_gradient2_used_error;

  /** @brief The lower bound on the total number of (n - 1) tuples
   *         pruned.
   */
  double n_pruned_;

  /** @brief The postponed estimate of the first component of the
   *         negative gradient.
   */
  double postponed_negative_gradient1_e;

  /** @brief The postponed lower bound change to the first component
   *         of the negative gradient.
   */
  double postponed_negative_gradient1_u;

  /** @brief The postponed used error for the first component of the
   *         negative gradient.
   */
  double postponed_negative_gradient1_used_error;

  /** @brief The postponed lower bound change to the first component
   *         of the positive gradient.
   */
  double postponed_positive_gradient1_l;

  /** @brief The postponed estimate of the first component of the
   *         positive gradient.
   */
  double postponed_positive_gradient1_e;

  /** @brief The postponed used error for the first component of the
   *         positive gradient.
   */
  double postponed_positive_gradient1_used_error;

  /** @brief The postponed estimate of the second component of the
   *         negative gradient.
   */
  Vector postponed_negative_gradient2_e;

  /** @brief The postponed lower bound change to the second component
   *         of the negative gradient.
   */
  Vector postponed_negative_gradient2_u;

  /** @brief The postponed used error for the second component of the
   *         negative gradient.
   */
  double postponed_negative_gradient2_used_error;

  /** @brief The postponed lower bound change to the second component
   *         of the positive gradient.
   */
  Vector postponed_positive_gradient2_l;

  /** @brief The postponed estimate of the second component of the
   *         positive gradient.
   */
  Vector postponed_positive_gradient2_e;

  /** @brief The postponed used error for the second component of the
   *         positive gradient.
   */
  double postponed_positive_gradient2_used_error;

  /** @brief The postponed (n - 1) tuples that were pruned.
   */
  double postponed_n_pruned_;
  
  /** @brief The lower bounds on the k-nearest neighbor distances.
   */
  Vector knn_dsqds_lower_bounds_;

  /** @brief The upper bounds on the k-farthest neighbor distances.
   */
  Vector kfn_dsqds_upper_bounds_;

  /** @brief Resets the postponed statistics to zero.
   */
  void SetZero() {
    postponed_negative_gradient1_e = 0;
    postponed_negative_gradient1_u = 0;
    postponed_negative_gradient1_used_error = 0;
    postponed_positive_gradient1_l = 0;
    postponed_positive_gradient1_e = 0;
    postponed_positive_gradient1_used_error = 0;
    postponed_negative_gradient2_e.SetZero();
    postponed_negative_gradient2_u.SetZero();
    postponed_negative_gradient2_used_error = 0;
    postponed_positive_gradient2_l.SetZero();
    postponed_positive_gradient2_e.SetZero();
    postponed_positive_gradient2_used_error = 0;
    postponed_n_pruned_ = 0;
  }

  /** @brief Initialize the statistics.
   */
  void Init() {

    coordinate_sum_.Init(3);    
    negative_gradient1_u = 0;
    negative_gradient1_used_error = 0;
    positive_gradient1_l = 0;
    positive_gradient1_used_error = 0;
    l1_norm_negative_gradient2_u = 0;
    negative_gradient2_used_error = 0;
    l1_norm_positive_gradient2_l = 0;
    positive_gradient2_used_error = 0;
    n_pruned_ = 0;

    postponed_negative_gradient1_e = 0;
    postponed_negative_gradient1_u = 0;
    postponed_negative_gradient1_used_error = 0;
    postponed_positive_gradient1_l = 0;
    postponed_positive_gradient1_e = 0;
    postponed_positive_gradient1_used_error = 0;
    postponed_negative_gradient2_e.Init(3);
    postponed_negative_gradient2_u.Init(3);
    postponed_negative_gradient2_used_error = 0;
    postponed_positive_gradient2_l.Init(3);
    postponed_positive_gradient2_e.Init(3);
    postponed_positive_gradient2_used_error = 0;
    postponed_n_pruned_ = 0;

    // Hard-coding for Axilrod-Teller: each point needs two nearest
    // neighbor distances.
    knn_dsqds_lower_bounds_.Init(1);
    kfn_dsqds_upper_bounds_.Init(1);
  }

  /** @brief The initialization for leaf stats.
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
    Init();

    coordinate_sum_.SetZero();
    l1_norm_coordinate_sum_ = 0;
    max_l1_norm_ = 0;
    for(index_t i = start; i < start + count; i++) {
      const double *point = dataset.GetColumnPtr(i);

      la::AddTo(dataset.n_rows(), point, coordinate_sum_.ptr());
      double l1_norm = 0;
      for(index_t d = 0; d < dataset.n_rows(); d++) {
	l1_norm += fabs(point[d]);
      }
      max_l1_norm_ = std::max(max_l1_norm_, l1_norm);
    }
    for(index_t d = 0; d < dataset.n_rows(); d++) {
      l1_norm_coordinate_sum_ += coordinate_sum_[d];
    }
    l1_norm_coordinate_sum_l_ = l1_norm_coordinate_sum_ - max_l1_norm_;

    SetZero();
  }

  /** @brief The initialization for non-leaf stats based on the stats
   *         owned by the child nodes.
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const MultibodyStat& left_stat, const MultibodyStat& right_stat) {
    Init();

    la::AddOverwrite(left_stat.coordinate_sum_, right_stat.coordinate_sum_,
		     &coordinate_sum_);
    l1_norm_coordinate_sum_ = left_stat.l1_norm_coordinate_sum_ +
      right_stat.l1_norm_coordinate_sum_;
    max_l1_norm_ = std::max(left_stat.max_l1_norm_, right_stat.max_l1_norm_);
    l1_norm_coordinate_sum_l_ = l1_norm_coordinate_sum_ - max_l1_norm_;
    SetZero();
  }

  ////////// Constructor/Destructor //////////

  /** @brief The default constructor.
   */
  MultibodyStat() {
  }

  /** @brief The default destructor.
   */
  ~MultibodyStat() {
  }

};

#endif
