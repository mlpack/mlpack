class MultiTreeQueryStat {

 public:

  MultiTreeQueryPostponed postponed;
    
  MultiTreeQuerySummary summary;
    
  double priority;
    
  Vector mean;
    
  index_t count;
    
  bool in_strata;

  double num_precomputed_tuples;
  
  OT_DEF_BASIC(MultiTreeQueryStat) {
    OT_MY_OBJECT(postponed);
    OT_MY_OBJECT(summary);
    OT_MY_OBJECT(priority);
    OT_MY_OBJECT(mean);
    OT_MY_OBJECT(count);
    OT_MY_OBJECT(in_strata);
    OT_MY_OBJECT(num_precomputed_tuples);
  }
  
 public:

  double SumOfPerDimensionVariances
  (const Matrix &dataset, index_t &start, index_t &count) {

    double total_variance = 0;
    for(index_t i = start; i < start + count; i++) {
      const double *point = dataset.GetColumnPtr(i);
      for(index_t d = 0; d < 3; d++) {
	total_variance += math::Sqr(point[d] - mean[d]);
      }
    }
    total_variance /= ((double) count);
    return total_variance;
  }

  void FinalPush(MultiTreeQueryStat &child_stat) {
    child_stat.postponed.ApplyPostponed(postponed);
  }
    
  void SetZero() {
    postponed.SetZero();
    summary.SetZero();
    priority = 0;
    mean.SetZero();
    in_strata = false;
    num_precomputed_tuples = 0;
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count_in) {
    postponed.Init();
    mean.Init(3);
    SetZero();
    count = count_in;

    // Compute the mean vector.
    for(index_t i = start; i < start + count; i++) {
      const double *point = dataset.GetColumnPtr(i);
      la::AddTo(3, point, mean.ptr());
    }
    la::Scale(3, 1.0 / ((double) count), mean.ptr());

    // Compute the priority of this node which is basically the
    // number of points times sum of per-dimension variances.
    double sum_of_per_dimension_variances = SumOfPerDimensionVariances
      (dataset, start, count);
    priority = count * sum_of_per_dimension_variances;
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count_in,
	    const MultiTreeQueryStat& left_stat, 
	    const MultiTreeQueryStat& right_stat) {
    postponed.Init();
    mean.Init(3);
    SetZero();
    count = count_in;

    la::ScaleOverwrite(left_stat.count, left_stat.mean, &mean);
    la::AddExpert(3, right_stat.count, right_stat.mean.ptr(), mean.ptr());
    la::Scale(3, 1.0 / ((double) count), mean.ptr());

    // Compute the priority of this node which is basically the
    // number of points times sum of per-dimension variances.
    double sum_of_per_dimension_variances = SumOfPerDimensionVariances
      (dataset, start, count);
    priority = count * sum_of_per_dimension_variances;
  }
    
  template<typename TKernelAux>
  void Init(const TKernelAux &kernel_aux_in) {
  }
    
  template<typename TBound, typename TKernelAux>
  void Init(const TBound &bounding_primitive,
	    const TKernelAux &kernel_aux_in) {
      
    // Reset the postponed quantities to zero.
    SetZero();
  }
};

class MultiTreeReferenceStat {

};
