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
#include "../../mlpack/core/kernels/lmetric.h"

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
// TO BE IMPLEMENTED
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MaxCosine(const Point& point) const {

  // cos A = cos <point, center_;
  // cos B = min cos<any_point, center_;
  // MaxCosine = max(cos(A+B), cos(A-B))

  return ;
}

/**
 * Calculates maximum bound-to-bound cosine.
 */
// TO BE IMPLEMENTED
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MaxCosine(const DConeBound& other) const {
  // cos A = cos <center_, other.center_;
  // cos B = cos <center_, any_point;
  // cos C = cos <other.center_, any_point;
  // MaxCosine = cos (A-(B+C)) if A > B+C
  //           = 1 if A < B+C

  return;
}

/**
 * Computes minimum cosine.
 */
// TO BE IMPLEMENTED
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MinCosine(const Point& point) const {

  // cos A = cos <point, center_;
  // cos B = min cos<any_point, center_;
  // MinCosine = min(cos(A+B), cos(A-B))

  return ;
}

/**
 * Computes minimum bonud-to-bound cosine.
 */
// TO BE IMPLEMENTED
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MinCosine(const DConeBound& other) const {

  // cos A = cos <center_, other.center_;
  // cos B = cos <center_, any_point;
  // cos C = cos <other.center_, any_point;
  // MinCosine = cos (A + (B + C)) if A + B + C <= pi
  //           = 0 otherwise

  return;
}

/**
 * Calculates minimum and maximum bound-to-bound cosine.
 */
// TO BE IMPLEMENTED
template<typename TMetric, typename TPoint>
DRange DConeBound<TMetric, TPoint>::RangeCosine(const DConeBound& other) const {
  double delta = MidCosine(other.center_);
  double sumradius = radius_ + other.radius_;
  return DRange(
      math::ClampNonNegative(delta - sumradius),
      delta + sumradius);
}

/**
 * Calculates closest-to-their-midpoint bounding box cosine,
 * i.e. calculates their midpoint and finds the minimum box-to-point
 * cosine.
 *
 * Equivalent to:
 * <code>
 * other.CalcMidpoint(&other_midpoint)
 * return MinCosineSqToPoint(other_midpoint)
 * </code>
 */
// TO BE IMPLEMENTED
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MinToMid(const DConeBound& other) const {
  double delta = MidCosine(other.center_) - radius_;
  return math::ClampNonNegative(delta);
}

/**
 * Computes minimax cosine, where the other node is trying to avoid me.
 */
template<typename TMetric, typename TPoint>
double DConeBound<TMetric, TPoint>::MinimaxCosine(const DConeBound& other) const {
  double delta = MidCosine(other.center_) + other.radius_ - radius_;
  return math::ClampNonNegative(delta);
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

