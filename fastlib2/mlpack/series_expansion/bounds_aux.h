#ifndef BOUNDS_AUX_H
#define BOUNDS_AUX_H

#include "fastlib/fastlib_int.h"
#include "tree/bounds.h"

namespace bounds_aux {
  
  template<int t_pow>
  void MaxDistanceSq(const DHrectBound<t_pow> &bound1, 
		     const DHrectBound<t_pow> &bound2,
		     Vector &furthest_point_in_bound1,
		     double *furthest_dsqd) {
		       
    int dim = furthest_point_in_bound1.length();
    *furthest_dsqd = 0;
    for (index_t d = 0; d < dim; d++) {
      
      const DRange &bound1_range = bound1.get(d);
      const DRange &bound2_range = bound2.get(d);
      
      double v1 = bound2_range.hi - bound1_range.lo;
      double v2 = bound1_range.hi - bound2_range.lo;
      double v;
      
      if(v1 > v2) {
	furthest_point_in_bound1[d] = bound1_range.lo;
	v = v1;
      }
      else {
	furthest_point_in_bound1[d] = bound1_range.hi;
	v = v2;
      }
      (*furthest_dsqd) += math::PowAbs<t_pow, 1>(v); // v is non-negative
    }
    *furthest_dsqd = math::Pow<2, t_pow>(*furthest_dsqd);
  }
};

#endif
