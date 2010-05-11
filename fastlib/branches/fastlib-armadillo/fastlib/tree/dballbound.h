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
/**
 * @file tree/dballbound.h
 *
 * Bounds that are useful for binary space partitioning trees.
 * Interface to a ball bound that works in arbitrary metric spaces.
 *
 * @experimental
 */

#ifndef TREE_DBALLBOUND_H
#define TREE_DBALLBOUND_H

#include "../la/matrix.h"
#include "../la/la.h"

#include "../math/math_lib.h"
#include "lmetric.h"

#include <armadillo>

/**
 * Ball bound that works in arbitrary metric spaces.
 *
 * See LMetric for an example metric template parameter.
 *
 * To initialize this, set the radius with @c set_radius
 * and set the point by initializing @c point() directly.
 */
template<typename TMetric = LMetric<2>, typename TPoint = arma::vec>
class DBallBound {
  public:
    typedef TPoint Point;
    typedef TMetric Metric;

  private:
    double radius_;
    TPoint center_;

    OBJECT_TRAVERSAL(DBallBound) {
      OT_OBJ(radius_);
      OT_OBJ(center_);
    }

  public:
    /***
     * Return the radius of the ball bound.
     */
    double radius() const { return radius_; }

    /***
     * Set the radius of the bound.
     */
    void set_radius(double d) { radius_ = d; }

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
     * Calculates minimum bound-to-point squared distance.
     */
    double MinDistance(const Point& point) const;
    double MinDistanceSq(const Point& point) const;

    /**
     * Calculates minimum bound-to-bound squared distance.
     */
    double MinDistance(const DBallBound& other) const;
    double MinDistanceSq(const DBallBound& other) const;

    /**
     * Computes maximum distance.
     */
    double MaxDistance(const Point& point) const;
    double MaxDistanceSq(const Point& point) const;

    /**
     * Computes maximum distance.
     */
    double MaxDistance(const DBallBound& other) const;
    double MaxDistanceSq(const DBallBound& other) const;

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    DRange RangeDistance(const DBallBound& other) const;
    DRange RangeDistanceSq(const DBallBound& other) const;

    /**
     * Calculates closest-to-their-midpoint bounding box distance,
     * i.e. calculates their midpoint and finds the minimum box-to-point
     * distance.
     *
     * Equivalent to:
     * <code>
     * other.CalcMidpoint(&other_midpoint)
     * return MinDistanceSqToPoint(other_midpoint)
     * </code>
     */
    double MinToMid(const DBallBound& other) const;
    double MinToMidSq(const DBallBound& other) const;

    /**
     * Computes minimax distance, where the other node is trying to avoid me.
     */
    double MinimaxDistance(const DBallBound& other) const;
    double MinimaxDistanceSq(const DBallBound& other) const;

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    double MidDistance(const DBallBound& other) const;
    double MidDistanceSq(const DBallBound& other) const;
    double MidDistance(const Point& point) const;
};

#include "dballbound_impl.h"

#endif
