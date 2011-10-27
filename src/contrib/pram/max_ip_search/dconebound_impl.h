/**
 * @file tree/dconebound_impl.h
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of DConeBound cone bound metric policy class.
 *
 * @experimental
 */

#ifndef TREE_DCONEBOUND_IMPL_H
#define TREE_DCONEBOUND_IMPL_H

// Awaiting transition
#include "cosine.h"

#include <armadillo>

/**
 * Determines if a point is within the cone bound.
 */
template<typename TMetric, typename TPoint>
bool DConeBound<TMetric, TPoint>::Contains(const Point& point) const {
  return MidCosine(point) >= radius_;
}

/**
 * Gets the center.
 *
 * Don't really use this directly.  This is only here for consistency
 * with DHrectBound, so it can plug in more directly if a "centroid"
 * is needed.
 */
template<typename TMetric, typename TPoint>
void DConeBound<TMetric, TPoint>::CalculateMidpoint(Point *centroid) const {
  (*centroid) = center_;
}

/**
 * Calculates maximum bound-to-point cosine.
 */
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MaxCosine(const Point& point) const {

  // cos A = cos <point, center_, A >= 0; 
  // cos B = min cos<any_point, center_, B >= 0; 
  // MaxCosine = 1 if cos A >= cos B or A <= B; 
  //           = cos (A - B) otherwise; 

  double cos_point_center = MidCosine(point);
  if (cos_point_center < radius_) {
    // cos (A - B) = cos A cos B + sin A sin B;
    double max_cosine = cos_point_center * radius_; 
    max_cosine += (std::sqrt(1 - cos_point_center * cos_point_center) 
		   * radius_conjugate_);

    return max_cosine;
  }

  return 1.0;
}

/**
 * Calculates maximum bound-to-bound cosine.
 */
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MaxCosine(const DConeBound& other) const {
  // cos A = cos <center_, other.center_, 0 < A < pi;
  // cos B = cos <center_, any_point, 0 < B < pi;
  // cos C = cos <other.center_, any_point, 0 < C < pi;
  // MaxCosine = cos (A-(B+C)) if A > B+C
  //           = 1 if A < B+C

  double cos_center_center = MidCosine(other.center_);

  if ((cos_center_center < radius_)
      && (cos_center_center < other.radius_)) {
    // A > B and A > C
    // Computing cos (A - B)
    double cos_cone_other_center = cos_center_center * radius_;
    cos_cone_other_center
      += (std::sqrt(1 - cos_center_center * cos_center_center) 
	  * radius_conjugate_);

    if (cos_cone_other_center < other.radius_) {
      // A - B > C
      // computing cos (A - B - C)
      double cos_cone_other_cone = cos_cone_other_center * other.radius_;
      cos_cone_other_cone
	+= (std::sqrt(1 - cos_cone_other_center * cos_cone_other_center) 
	    * other.radius_conjugate_);

      return cos_cone_other_cone;
    }
  } 
  return 1.0;
}

/**
 * Computes minimum cosine.
 */
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MinCosine(const Point& point) const {

  // cos A = cos <point, center_, 0 < A < pi;
  // cos B = min cos<any_point, center_, 0 < B < pi;
  // MinCosine = cos (A + B) if cos (pi - A) < cos B
  //           = -1.0 otherwise

  double cos_point_center = MidCosine(point);
  if (-cos_point_center < radius_) {
    // cos (A + B) = cos A cos B - sin A sin B;
    double min_cosine = cos_point_center * radius_; 
    min_cosine -= (std::sqrt(1 - cos_point_center * cos_point_center) 
		   * radius_conjugate_);

    return min_cosine;
  }

  return -1.0;
}

/**
 * Computes minimum bound-to-bound cosine.
 */
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MinCosine(const DConeBound& other) const {

  // cos A = cos <center_, other.center_, 0 < A < pi;
  // cos B = cos <center_, any_point, 0 < B < pi;
  // cos C = cos <other.center_, any_point, 0 < C < pi;
  // MinCosine = cos (A + B + C) if A + B + C < pi;
  //           = -1.0 otherwise

  double cos_center_center = MidCosine(other.center_);

  if ((-cos_center_center < radius_)
      && (-cos_center_center < other.radius_)) {
    // A + B < pi and A + C < pi
    // Computing cos (A + B)
    double cos_far_cone_other_center = cos_center_center * radius_;
    cos_far_cone_other_center
      -= (std::sqrt(1 - cos_center_center * cos_center_center) 
	  * radius_conjugate_);

    if (-cos_far_cone_other_center < other.radius_) {
      // A + B + C < pi
      // computing cos (A + B + C)
      double cos_far_cone_other_far_cone
	= cos_far_cone_other_center * other.radius_;
      cos_far_cone_other_far_cone
	-= (std::sqrt(1 - cos_far_cone_other_center 
		      * cos_far_cone_other_center) 
	    * other.radius_conjugate_);

      return cos_far_cone_other_far_cone;
    }
  } 
  return -1.0;
}

/**
 * Calculates minimum and maximum bound-to-bound cosine.
 */
template<typename TMetric, typename TPoint>
mlpack::math::Range DConeBound<TMetric, TPoint>::RangeCosine(const DConeBound& other) const {
  return mlpack::math::Range(MinCosine(other), MaxCosine(other));
}

/**
 * Calculates closest-to-their-midpoint cosine,
 * i.e. calculates their midpoint and finds the maximum cone-to-point
 * cosine.
 *
 * Equivalent to:
 * <code>
 * return MaxCosine(&other.center_)
 * </code>
 */
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MaxToMid(const DConeBound& other) const {

  double cos_center_center = MidCosine(other.center_);

  if (cos_center_center < radius_) {
    // A > B
    // MaxToMid = cos (A - B)
    double max_cos_other_center = cos_center_center * radius_; 
    max_cos_other_center 
      += (std::sqrt(1 - cos_center_center * cos_center_center) 
	  * radius_conjugate_);
    
    return max_cos_other_center;
  }

  return 1.0;
}

/**
 * Computes minimax cosine, where the other node is trying to avoid me.
 */
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MinimaxCosine(const DConeBound& other) const {
  // cos A = cos <center_, other.center_, 0 < A < pi;
  // cos B = cos <center_, any_point, 0 < B < pi;
  // cos C = cos <other.center_, any_point, 0 < C < pi;
  // MinimaxCosine = -1.0 if cos (pi - (A - B)) >= cos C;
  //               = 1.0 if cos (A + C) > cos B
  //               = cos (A - B + C) otherwise

  double cos_center_center = MidCosine(other.center_);

  if (cos_center_center < radius_) {
    // A > B
    // computing cos (A - B)
    double max_cos_other_center = cos_center_center * radius_; 
    max_cos_other_center 
      += (std::sqrt(1 - cos_center_center * cos_center_center) 
	  * radius_conjugate_);

    if (-max_cos_other_center < other.radius_) {
      // A - B + C < pi
      // computing cos (A - B + C)
      double minimax_cosine = max_cos_other_center * other.radius_;
      minimax_cosine 
	-= (std::sqrt(1 - max_cos_other_center * max_cos_other_center) 
	    * other.radius_conjugate_);

      return minimax_cosine;
    } else {
      // A - B + C >= pi
      return -1.0;
    }
  } else {
    // B > A, check if B > A + C
    // computing cos (A + C)
    double min_cos_center_other_cone = cos_center_center * other.radius_;
    min_cos_center_other_cone 
      -= (std::sqrt(1 - cos_center_center * cos_center_center) 
	  * other.radius_conjugate_);

    if (min_cos_center_other_cone < radius_) {
      // A + C > B
      // computing cos (A - B + C)
      double minimax_cosine = min_cos_center_other_cone * radius_;
      minimax_cosine
	+= (std::sqrt(1 - min_cos_center_other_cone 
		      * min_cos_center_other_cone) * radius_conjugate_);

      return minimax_cosine;
    }
    
    return 1.0;
  }
}

/**
 * Calculates midpoint-to-midpoint bounding box cosine.
 */
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MidCosine(const DConeBound& other) const {
  return MidCosine(other.center_);
}

template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MidCosine(const Point& point) const {
  return Metric::Evaluate(center_, point);
}

#endif

