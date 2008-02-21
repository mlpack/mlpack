#ifndef LPR_UTIL_H
#define LPR_UTIL_H

class LprUtil {
  
 public:

  template<typename QueryTree, typename ReferenceTree>
  static void SqdistAndKernelRanges_
  (QueryTree *qnode, ReferenceTree *rnode, DRange &dsqd_range, 
   DRange &kernel_value_range) {
    
    // The following assumes that you are using a monotonically
    // decreasing kernel!
    dsqd_range = qnode->bound().RangeDistanceSq(rnode->bound());
    kernel_value_range.lo =
      rnode->stat().min_bandwidth_kernel.EvalUnnormOnSq(dsqd_range.hi);
    kernel_value_range.hi =
      rnode->stat().max_bandwidth_kernel.EvalUnnormOnSq(dsqd_range.lo);
  }
};

#endif
