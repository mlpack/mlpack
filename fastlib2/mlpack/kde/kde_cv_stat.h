#ifndef KDE_CV_STAT_H
#define KDE_CV_STAT_H

template<typename TKernel>
class VKdeCVStat {
 public:

  /** @brief The minimum bandwidth among the points owned by this
   *         node.
   */
  TKernel min_bandwidth_kernel_;

  /** @brief The maximum bandwidth among the points owned by this
   *         node.
   */
  TKernel max_bandwidth_kernel_;

  /** @brief The weight sum of the points owned by this node.
   */
  double weight_sum_;

  /** @brief Gets the weight sum.
   */
  double get_weight_sum() {
    return weight_sum_;
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const VKdeCVStat& left_stat, const VKdeCVStat& right_stat) {
  }
  
  VKdeCVStat() {
  }
    
  ~VKdeCVStat() {
  }
    
};

template<typename TKernelAux>
class KdeCVStat {
 public:

  /** @brief The far field expansion created by the reference points
   *         in this node.
   */
  typename TKernelAux::TFarFieldExpansion first_farfield_expansion_;
  
  /** @brief The far field expansion
   */
  typename TKernelAux::TFarFieldExpansion second_farfield_expansion_;

  /** @brief The subspace associated with this node.
   */
  SubspaceStat subspace_;

  /** @brief Gets the weight sum.
   */
  double get_weight_sum() {
    return first_farfield_expansion_.get_weight_sum();
  }
    
  void Init(const TKernelAux &first_ka, const TKernelAux &second_ka) {
    first_farfield_expansion_.Init(first_ka);
    second_farfield_expansion_.Init(second_ka);
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
    subspace_.Init(dataset, start, count);
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const KdeCVStat& left_stat, const KdeCVStat& right_stat) {
    subspace_.Init(dataset, start, count, left_stat.subspace_,
		   right_stat.subspace_);
  }
    
  void Init(const Vector& center, const TKernelAux &first_ka,
	    const TKernelAux &second_ka) {
    first_farfield_expansion_.Init(center, first_ka);
    second_farfield_expansion_.Init(center, second_ka);
  }
    
  KdeCVStat() { }
    
  ~KdeCVStat() { }
    
};

#endif
