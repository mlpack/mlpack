/**
 * @file dconebound.h
 *
 * Bounds that are useful for binary space partitioning trees.
 * Interface to a cone bound that works with cosine 
 * similarity only (for now)
 *
 * @experimental
 */

#ifndef TREE_DCONEBOUND_H
#define TREE_DCONEBOUND_H

#include "mlpack/core/math/range.hpp"

// Awaiting transition
#include "cosine.h"

#include <armadillo>

/**
 * Cone bound that works in arbitrary metric spaces.
 *
 * See LMetric for an example metric template parameter.
 *
 * To initialize this, set the radius with @c set_radius
 * and set the point by initializing @c point() directly.
 */
template<typename TMetric = Cosine, typename TPoint = arma::vec>
class DConeBound {
  public:
    typedef TPoint Point;
    typedef TMetric Metric;

  private:
    double radius_;
    double radius_conjugate_;
    TPoint center_;

  public:
    /***
     * Return the radius of the cone bound.
     */
    double radius() const { return radius_; }
  double radius_conjugate() const { return radius_conjugate_; }

    /***
     * Set the radius of the bound.
     */
    void set_radius(double d) { 
      radius_ = d;
      radius_conjugate_ = std::sqrt(1 - d*d);
    }

    /***
     * Return the center point.
     */
    const TPoint& center() const { return center_; }

    /***
     * Return the center point.
     */
    TPoint& center() { return center_; }

    /**
     * Determines if a point is within this bound.
     */
    bool Contains(const Point& point) const;

    /**
     * Gets the center.
     *
     * Don't really use this directly.  This is only here for consistency
     * with DHrectBound, so it can plug in more directly if a "centroid"
     * is needed.
     */
    void CalculateMidpoint(Point *centroid) const;

    /**
     * Calculates maximum bound-to-point cosine.
     */
    double MaxCosine(const Point& point) const;

    /**
     * Calculates maximum bound-to-bound cosine.
     */
    double MaxCosine(const DConeBound& other) const;

    /**
     * Computes maximum distance.
     */
    double MinCosine(const Point& point) const;

    /**
     * Computes maximum distance.
     */
    double MinCosine(const DConeBound& other) const;

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    mlpack::math::Range RangeCosine(const DConeBound& other) const;

    /**
     * Calculates closest-to-their-midpoint bounding box distance,
     * i.e. calculates their midpoint and finds the minimum box-to-point
     * distance.
     *
     * Equivalent to:
     * <code>
     * other.CalcMidpoint(&other_midpoint)
     * return MaxCosineToPoint(other_midpoint)
     * </code>
     */
    double MaxToMid(const DConeBound& other) const;

    /**
     * Computes minimax distance, where the other node is trying to avoid me.
     */
    double MinimaxCosine(const DConeBound& other) const;

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    double MidCosine(const DConeBound& other) const;
    double MidCosine(const Point& point) const;
};

#include "dconebound_impl.h"

#endif
