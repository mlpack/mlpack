#ifndef BOUNDS_AUX_H
#define BOUNDS_AUX_H

#include "fastlib/fastlib_int.h"

namespace bounds_aux {
  
  template<int t_pow>
  void MaxDistanceSq(const DHrectBound<t_pow> &bound1, 
		     const DHrectBound<t_pow> &bound2,
		     Vector &furthest_point_in_bound1,
		     double &furthest_dsqd) {
		       
    int dim = furthest_point_in_bound1.length();
    furthest_dsqd = 0;

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
      furthest_dsqd += math::PowAbs<t_pow, 1>(v); // v is non-negative
    }
    furthest_dsqd = math::Pow<2, t_pow>(furthest_dsqd);
  }

  template<int t_pow>
  void MaxDistanceSq(const DHrectBound<t_pow> &bound1, 
		     Vector &bound2_centroid,
		     Vector &furthest_point_in_bound1,
		     double &furthest_dsqd) {
		       
    int dim = furthest_point_in_bound1.length();
    furthest_dsqd = 0;

    for (index_t d = 0; d < dim; d++) {
      
      const DRange &bound1_range = bound1.get(d);
      
      double v1 = bound2_centroid[d] - bound1_range.lo;
      double v2 = bound1_range.hi - bound2_centroid[d];
      double v;
      
      if(v1 > v2) {
	furthest_point_in_bound1[d] = bound1_range.lo;
	v = v1;
      }
      else {
	furthest_point_in_bound1[d] = bound1_range.hi;
	v = v2;
      }
      furthest_dsqd += math::PowAbs<t_pow, 1>(v); // v is non-negative
    }
    furthest_dsqd = math::Pow<2, t_pow>(furthest_dsqd);
  }

  template<int t_pow, typename TVector>
  void MaxDistanceSq(const DBallBound < LMetric<t_pow>, TVector > &bound1, 
		     Vector &bound2_centroid,
		     Vector &furthest_point_in_bound1,
		     double &furthest_dsqd) {
		       
    furthest_dsqd = 0;
    
    // First compute the distance between the centroid of the bounding
    // ball and the given point.
    double distance = 
      LMetric<t_pow>::Distance(bound1.center(), bound2_centroid);
    
    // Compute the unit vector that has the same direction as the
    // vector pointing from the given point to the bounding ball
    // center.
    Vector unit_vector;
    la::SubInit(bound2_centroid, bound1.center(), &unit_vector);
    la::Scale(1.0 / distance, &unit_vector);

    furthest_point_in_bound1.CopyValues(bound1.center());
    la::AddExpert(bound1.radius(), unit_vector, &furthest_point_in_bound1);
    
    furthest_dsqd = math::Pow<2, t_pow>
      (la::RawLMetric<t_pow>(bound2_centroid.length(), 
			     furthest_point_in_bound1.ptr(), 
			     bound2_centroid.ptr()));
  }
};

#endif
