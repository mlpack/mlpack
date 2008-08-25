#ifndef KDE_STAT_H
#define KDE_STAT_H

template<typename TKernelAux>
class KdeStat {
 public:
  
  /** @brief The lower bound on the densities for the query points
   *         owned by this node.
   */
  double mass_l_;
  
  /** @brief The upper bound on the densities for the query points
   *         owned by this node
   */
  double mass_u_;
  
  /** @brief Upper bound on the used error for the query points
   *         owned by this node.
   */
  double used_error_;
  
  /** @brief Lower bound on the number of reference points taken
   *         care of for query points owned by this node.
   */
  double n_pruned_;

  /** @brief The lower bound on the maximum kernel value.
   */
  double max_kernel_value_l_;

  /** @brief The lower bound offset passed from above.
   */
  double postponed_l_;
  
  /** @brief Stores the portion pruned by finite difference.
   */
  double postponed_e_;

  /** @brief The upper bound offset passed from above.
   */
  double postponed_u_;

  /** @brief The total amount of error used in approximation for all query
   *         points that must be propagated downwards.
   */
  double postponed_used_error_;

  /** @brief The number of reference points that were taken care of
   *         for all query points under this node; this information
   *         must be propagated downwards.
   */
  double postponed_n_pruned_;

  /** @brief The far field expansion created by the reference points
   *         in this node.
   */
  typename TKernelAux::TFarFieldExpansion farfield_expansion_;
    
  /** @brief The local expansion stored in this node.
   */
  typename TKernelAux::TLocalExpansion local_expansion_;
    
  /** @brief The subspace associated with this node.
   */
  SubspaceStat subspace_;
  
  /** @brief Adds the other postponed contributions.
   */
  void AddPostponed(const KdeStat& parent_stat) {
    postponed_l_ += parent_stat.postponed_l_;
    postponed_e_ += parent_stat.postponed_e_;
    postponed_u_ += parent_stat.postponed_u_;
    postponed_used_error_ += parent_stat.postponed_used_error_;
    postponed_n_pruned_ += parent_stat.postponed_n_pruned_;
  }

  /** @brief Clears the postponed contributions.
   */
  void ClearPostponed() {      
    postponed_l_ = 0;
    postponed_e_ = 0;
    postponed_u_ = 0;
    postponed_used_error_ = 0;
    postponed_n_pruned_ = 0;      
  }

  /** @brief Refines the bound statistics.
   */
  void RefineBoundStatistics(const KdeStat &left_child_stat,
			     const KdeStat &right_child_stat) {
    mass_l_ = std::min(left_child_stat.mass_l_ + left_child_stat.postponed_l_,
		       right_child_stat.mass_l_ +
		       right_child_stat.postponed_l_);
    mass_u_ = std::max(left_child_stat.mass_u_ + left_child_stat.postponed_u_,
		       right_child_stat.mass_u_ +
		       right_child_stat.postponed_u_);
    used_error_ =
      std::max(left_child_stat.used_error_ +
	       left_child_stat.postponed_used_error_,
	       right_child_stat.used_error_ +
	       right_child_stat.postponed_used_error_);
    n_pruned_ = 
      std::min(left_child_stat.n_pruned_ + left_child_stat.postponed_n_pruned_,
	       right_child_stat.n_pruned_ + right_child_stat.n_pruned_);
    max_kernel_value_l_ =
      std::min(left_child_stat.max_kernel_value_l_,
	       right_child_stat.max_kernel_value_l_);
  }

  /** @brief Resets the bound statistics.
   */
  void ResetBoundStatistics() {
    mass_l_ = DBL_MAX;
    mass_u_ = -DBL_MAX;
    used_error_ = 0;
    n_pruned_ = DBL_MAX;
    max_kernel_value_l_ = 0;
  }

  /** @brief Initialize the statistics.
   */
  void Init() {
    mass_l_ = 0;
    mass_u_ = 0;
    used_error_ = 0;
    n_pruned_ = 0;
    max_kernel_value_l_ = 0;
     
    postponed_l_ = 0;
    postponed_e_ = 0;
    postponed_u_ = 0;
    postponed_used_error_ = 0;
    postponed_n_pruned_ = 0;
  }
    
  void Init(const TKernelAux &ka) {
    farfield_expansion_.Init(ka);
    local_expansion_.Init(ka);
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
    Init();
    subspace_.Init(dataset, start, count);
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const KdeStat& left_stat,
	    const KdeStat& right_stat) {
    Init();
    subspace_.Init(dataset, start, count, left_stat.subspace_,
		   right_stat.subspace_);
  }
    
  void Init(const Vector& center, const TKernelAux &ka) {
      
    farfield_expansion_.Init(center, ka);
    local_expansion_.Init(center, ka);
    Init();
  }
    
  KdeStat() { }
    
  ~KdeStat() { }
    
};

#endif
