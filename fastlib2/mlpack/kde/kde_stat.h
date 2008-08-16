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

  /** @brief Clears the postponed contributions.
   */
  void ClearPostponed() {      
    postponed_l_ = 0;
    postponed_e_ = 0;
    postponed_u_ = 0;
    postponed_used_error_ = 0;
    postponed_n_pruned_ = 0;      
  }

  /** @brief Initialize the statistics.
   */
  void Init() {
    mass_l_ = 0;
    mass_u_ = 0;
    used_error_ = 0;
    n_pruned_ = 0;     
     
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
