#ifndef BOUNDS_AUX_H
#define BOUNDS_AUX_H

#include "fastlib/fastlib.h"

class bounds_aux {
  
 public:
  template<int t_pow>
  static void MaxDistanceSq(const DHrectBound<t_pow> &bound1, 
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
  static void MaxDistanceSq(const DHrectBound<t_pow> &bound1, 
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
  static void MaxDistanceSq(const DBallBound < LMetric<t_pow>, TVector > &bound1, 
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

  /** @brief Returns the maximum side length of the bounding box that
   *         encloses the given ball bound. That is, twice the radius
   *         of the given ball bound.
   */
  template<int t_pow, typename TVector>
  static double MaxSideLengthOfBoundingBox
  (const DBallBound < LMetric<t_pow>, TVector > &ball_bound) {
    return ball_bound.radius() * 2;
  }
  
  /** @brief Returns the maximum side length of the bounding box that
   *         encloses the given bounding box. That is, its maximum
   *         side length.
   */
  template<int t_pow>
  static double MaxSideLengthOfBoundingBox(const DHrectBound<t_pow> &bound) {

    double max_length = 0;

    for(index_t d = 0; d < bound.dim(); d++) {
      const DRange &range = bound.get(d);
      max_length = std::max(max_length, range.width());
    }
    return max_length;
  }
  
  /** @brief Returns the maximum distance between two bound types in
   *         L1 sense.
   */
  template<int t_pow, typename TVector>
  static double MaxL1Distance
  (const DBallBound < LMetric<t_pow>, TVector > &ball_bound1,
   const DBallBound < LMetric<t_pow>, TVector > &ball_bound2,
   int *dimension) {
    
    const Vector &center1 = ball_bound1.center();
    const Vector &center2 = ball_bound2.center();
    int dim = ball_bound1.center().length();
    double l1_distance = 0;
    for(index_t d = 0; d < dim; d++) {
      l1_distance += fabs(center1[d] - center2[d]);
    }
    l1_distance += ball_bound1.radius() + ball_bound2.radius();
    *dimension = center1.length();
    return l1_distance;
  }

  /** @brief Returns the maximum distance between two bound types in
   *         L1 sense.
   */
  template<int t_pow>
  static double MaxL1Distance(const DHrectBound<t_pow> &bound1,
			      const DHrectBound<t_pow> &bound2, int *dimension) {

    double farthest_distance_manhattan = 0;
    for(index_t d = 0; d < bound1.dim(); d++) {
      const DRange &range1 = bound1.get(d);
      const DRange &range2 = bound2.get(d);
      double bound1_centroid_coord = range1.lo + range1.width() / 2;

      farthest_distance_manhattan =
	max(farthest_distance_manhattan,
	    max(fabs(bound1_centroid_coord - range2.lo),
		fabs(bound1_centroid_coord - range2.hi)));
    }
    *dimension = bound1.dim();
    return farthest_distance_manhattan;
  }
};

#endif
