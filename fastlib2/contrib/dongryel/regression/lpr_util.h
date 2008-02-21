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

  template<typename QueryTree, typename ReferenceTree>
  static void BestQueryNodePartners
  (ReferenceTree *nd, QueryTree *nd1, QueryTree *nd2,
   QueryTree **partner1, QueryTree **partner2) {
    
    double d1 = nd->bound().MinDistanceSq(nd1->bound());
    double d2 = nd->bound().MinDistanceSq(nd2->bound());

    if(d1 <= d2) {
      *partner1 = nd1;
      *partner2 = nd2;
    }
    else {
      *partner1 = nd2;
      *partner2 = nd1;
    }
  }

  template<typename QueryTree, typename ReferenceTree>
  static void BestReferenceNodePartners
  (QueryTree *nd, ReferenceTree *nd1, ReferenceTree *nd2,
   ReferenceTree **partner1, ReferenceTree **partner2) {
    
    double d1 = nd->bound().MinDistanceSq(nd1->bound());
    double d2 = nd->bound().MinDistanceSq(nd2->bound());

    if(d1 <= d2) {
      *partner1 = nd1;
      *partner2 = nd2;
    }
    else {
      *partner1 = nd2;
      *partner2 = nd1;
    }
  }
};

#endif
