/**
 * @file dcosinebound.h
 *
 * Bounds that are useful for binary space partitioning trees.
 * Interface to a cosine bound that works with cosine 
 * similarity only (for now)
 *
 * @experimental
 */

#ifndef TREE_DCOSINEBOUND_H
#define TREE_DCOSINEBOUND_H

#include "mlpack/core/math/range.hpp"

// Awaiting transition
#include "cosine.h"

#include <armadillo>

/**
 * Cosine bound that works in arbitrary metric spaces.
 *
 * See LMetric for an example metric template parameter.
 *
 * To initialize this, set the radius with @c set_radius
 * and set the point by initializing @c point() directly.
 */
template<typename TMetric = Cosine, typename TPoint = arma::vec>
class DCosineBound {
public:
  typedef TPoint Point;
  typedef TMetric Metric;

private:
  double rad_min_;
  double rad_max_;
  double radius_;
  TPoint center_;

public:
  /***
   * Return the radius of the cosine bound.
   */
  double rad_min() const { return rad_min_; }
  double rad_max() const { return rad_max_; }
  double radius() const { return radius_; }

  /***
   * Set the radius of the bound.
   */
  void set_radius(double rad_min, double rad_max) { 

    rad_min_ = rad_min;
    rad_max_ = rad_max;
    radius_ = rad_max - rad_min;
  }

  /***
   * Return the center point.
   */
  const TPoint& center() const { return center_; }

  /***
   * Return the center point.
   */
  TPoint& center() { return center_; }

  // IMPLEMENT THESE LATER IN CASE THIS WORKS OUT
  //   /**
  //    * Determines if a point is within this bound.
  //    */
  //   bool Contains(const Point& point) const;

  //   /**
  //    * Gets the center.
  //    *
  //    * Don't really use this directly.  This is only here for consistency
  //    * with DHrectBound, so it can plug in more directly if a "centroid"
  //    * is needed.
  //    */
  //   void CalculateMidpoint(Point *centroid) const;

  //   /**
  //    * Calculates maximum bound-to-point cosine.
  //    */
  //   double MaxCosine(const Point& point) const;

  //   /**
  //    * Calculates maximum bound-to-bound cosine.
  //    */
  //   double MaxCosine(const DCosineBound& other) const;

  //   /**
  //    * Computes maximum distance.
  //    */
  //   double MinCosine(const Point& point) const;

  //   /**
  //    * Computes maximum distance.
  //    */
  //   double MinCosine(const DCosineBound& other) const;

  //   /**
  //    * Calculates minimum and maximum bound-to-bound squared distance.
  //    *
  //    * Example: bound1.MinDistanceSq(other) for minimum squared distance.
  //    */
  //   mlpack::math::Range RangeCosine(const DCosineBound& other) const;

  //   /**
  //    * Calculates closest-to-their-midpoint bounding box distance,
  //    * i.e. calculates their midpoint and finds the minimum box-to-point
  //    * distance.
  //    *
  //    * Equivalent to:
  //    * <code>
  //    * other.CalcMidpoint(&other_midpoint)
  //    * return MaxCosineToPoint(other_midpoint)
  //    * </code>
  //    */
  //   double MaxToMid(const DCosineBound& other) const;

  //   /**
  //    * Computes minimax distance, where the other node is trying to avoid me.
  //    */
  //   double MinimaxCosine(const DCosineBound& other) const;

  //   /**
  //    * Calculates midpoint-to-midpoint bounding box distance.
  //    */
  //   double MidCosine(const DCosineBound& other) const;
  //   double MidCosine(const Point& point) const;

}; // DCosineBound

// #include "dcosinebound_impl.h"

#endif
