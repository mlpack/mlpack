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

  /** @brief The maximum negative gradient (first component).
   */
  double negative_gradient1_u;

  /** @brief The minimum positive gradient (first component).
   */
  double positive_gradient1_l;
  
  /** @brief The maximum negative gradient (second component).
   */
  Vector negative_gradient2_u;

  /** @brief The minimum positive gradient (second component).
   */
  Vector positive_gradient2_l;

  /** @brief The postponed estimate of the first component of the
   *         negative gradient.
   */
  double postponed_negative_gradient1_e;

  /** @brief The postponed lower bound change to the first component
   *         of the negative gradient.
   */
  double postponed_negative_gradient1_u;

  /** @brief The postponed lower bound change to the first component
   *         of the positive gradient.
   */
  double postponed_positive_gradient1_l;

  /** @brief The postponed estimate of the first component of the
   *         positive gradient.
   */
  double postponed_positive_gradient1_e;

  /** @brief The postponed estimate of the second component of the
   *         negative gradient.
   */
  Vector postponed_negative_gradient2_e;

  /** @brief The postponed lower bound change to the second component
   *         of the negative gradient.
   */
  Vector postponed_negative_gradient2_u;

  /** @brief The postponed lower bound change to the second component
   *         of the positive gradient.
   */
  Vector postponed_positive_gradient2_l;

  /** @brief The postponed estimate of the second component of the
   *         positive gradient.
   */
  Vector postponed_positive_gradient2_e;

  /** @brief Resets the statistics to zero.
   */
  void SetZero() {
    negative_gradient1_u = 0;
    positive_gradient1_l = 0;
    negative_gradient2_u.SetZero();
    positive_gradient2_l.SetZero();

    postponed_negative_gradient1_e = 0;
    postponed_negative_gradient1_u = 0;
    postponed_positive_gradient1_l = 0;
    postponed_positive_gradient1_e = 0;
    postponed_negative_gradient2_e.SetZero();
    postponed_negative_gradient2_u.SetZero();
    postponed_positive_gradient2_l.SetZero();
    postponed_positive_gradient2_e.SetZero();
  }

  /** @brief Initialize the statistics.
   */
  void Init() {

    coordinate_sum_.Init(3);
    negative_gradient1_u = 0;
    positive_gradient1_l = 0;
    negative_gradient2_u.Init(3);
    positive_gradient2_l.Init(3);

    postponed_negative_gradient1_e = 0;
    postponed_negative_gradient1_u = 0;
    postponed_positive_gradient1_l = 0;
    postponed_positive_gradient1_e = 0;
    postponed_negative_gradient2_e.Init(3);
    postponed_negative_gradient2_u.Init(3);
    postponed_positive_gradient2_l.Init(3);
    postponed_positive_gradient2_e.Init(3);
  }

  /** @brief The initialization for leaf stats.
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
    Init();

    coordinate_sum_.SetZero();
    l1_norm_coordinate_sum_ = 0;
    for(index_t i = start; i < start + count; i++) {
      la::AddTo(dataset.n_rows(), dataset.GetColumnPtr(i), 
		coordinate_sum_.ptr());
    }
    for(index_t d = 0; d < dataset.n_rows(); d++) {
      l1_norm_coordinate_sum_ += coordinate_sum_[d];
    }
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
