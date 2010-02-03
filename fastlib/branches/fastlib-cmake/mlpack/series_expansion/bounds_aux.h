/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
#ifndef BOUNDS_AUX_H
#define BOUNDS_AUX_H

#include "fastlib/fastlib.h"
//#include "fastlib/fastlib_int.h"

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

  /** @brief Returns the maximum side length of the bounding box that
   *         encloses the given ball bound. That is, twice the radius
   *         of the given ball bound.
   */
  template<int t_pow, typename TVector>
  double MaxSideLengthOfBoundingBox
  (const DBallBound < LMetric<t_pow>, TVector > &ball_bound) {
    return ball_bound.radius() * 2;
  }
  
  /** @brief Returns the maximum side length of the bounding box that
   *         encloses the given bounding box. That is, its maximum
   *         side length.
   */
  template<int t_pow>
  double MaxSideLengthOfBoundingBox(const DHrectBound<t_pow> &bound) {

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
  double MaxL1Distance
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
  double MaxL1Distance(const DHrectBound<t_pow> &bound1,
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

  /** @brief The comparison function used for quick sort
   */
  static int qsort_compar_(const void *a, const void *b) {
    
    index_t a_dereferenced = *((index_t *) a);
    index_t b_dereferenced = *((index_t *) b);
    
    if(a_dereferenced < b_dereferenced) {
      return -1;
    }
    else if(a_dereferenced > b_dereferenced) {
      return 1;
    }
    else {
      return 0;
    }
  }

  double RandomizedDistanceSqEuclidean(index_t length, const double *a, 
				       const double *b) {
    
    int num_samples = (int) sqrt(length);
    double sample_mean = 0;
    ArrayList<index_t> dimension_indices;
    dimension_indices.Init(num_samples);
    for(index_t i = 0; i < num_samples; i++) {
      dimension_indices[i] = math::RandInt(0, length);
    }
    qsort(dimension_indices.begin(), num_samples, sizeof(index_t), 
	  &qsort_compar_);
    
    for(index_t i = 0; i < num_samples; i++) {      
      sample_mean += math::Sqr(a[dimension_indices[i]] - 
			       b[dimension_indices[i]]);
    }

    sample_mean /= ((double) num_samples);    
    return sample_mean * length;
  }

};

#endif
