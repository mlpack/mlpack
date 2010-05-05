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
 * @file tree/dhrectbound.h
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the interface for the DHrectBound policy, which
 * implements a hyperrectangle bound.
 *
 * @experimental
 */

#ifndef TREE_DHRECTBOUND_H
#define TREE_DHRECTBOUND_H

#include "../la/matrix.h"
#include "../la/la.h"

#include "../math/math_lib.h"

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<int t_pow = 2>
class DHrectBound {
  public:
    static const int PREFERRED_POWER = t_pow;

  private:
    DRange *bounds_;
    index_t dim_;

    OBJECT_TRAVERSAL(DHrectBound) {
      OT_OBJ(dim_);
      OT_ALLOC(bounds_, dim_);
    };

  public:
    /**
     * Initializes to specified dimensionality with each dimension the empty
     * set.
     */
    void Init(index_t dimension);

    /**
     * Makes this (uninitialized) box the average of the two arguments, 
     * i.e. the max and min of each range is the average of the maxes and mins 
     * of the arguments.  
     *
     * Added by: Bill March, 5/7
     */
    void AverageBoxesInit(const DHrectBound& box1, const DHrectBound& box2);

    /**
     * Resets all dimensions to the empty set.
     */
    void Reset();

    /**
     * Determines if a point is within this bound.
     */
    bool Contains(const Vector& point) const;
    bool Contains(const Vector& point, const Vector& box) const; 

    /** Gets the dimensionality */
    index_t dim() const { return dim_; }

    /**
     * Gets the range for a particular dimension.
     */
    const DRange& get(index_t i) const;

    /**
     * Calculates the maximum distance within the rectangle
     */
    double CalculateMaxDistanceSq() const;

    /** Calculates the midpoint of the range */
    void CalculateMidpoint(Vector *centroid) const;
    /** Calculates the midpoint of the range, assuming centroid is
      * already initialized to the right size */
    void CalculateMidpointOverwrite(Vector *centroid) const;

    /**
     * Calculates minimum bound-to-point squared distance.
     */
    double MinDistanceSq(const double *mpoint) const;

    /**
     * Calculates minimum bound-to-bound squared distance, with
     * an offset between their respective coordinate systems.
     */
    double MinDistanceSq(const DHrectBound& other, const Vector& offset) const;

    /**
     * Calculates minimum bound-to-point squared distance.
     */
    double MinDistanceSq(const Vector& point) const;

    /**
     * Calculates minimum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    double MinDistanceSq(const DHrectBound& other) const;

    /**
     * Calculates maximum bound-to-point squared distance.
     */
    double MaxDistanceSq(const Vector& point) const;

    /**
     * Calculates maximum bound-to-point squared distance.
     */
    double MaxDistanceSq(const double *point) const;

    /**
     * Computes maximum distance.
     */
    double MaxDistanceSq(const DHrectBound& other) const;

    /**
     * Computes maximum distance with offset
     */
    double MaxDistanceSq(const DHrectBound& other, const Vector& offset) const;

    /**
     * Computes minimum distance between boxes in periodic coordinate system
     */
    double PeriodicMinDistanceSq(const DHrectBound& other, const Vector& box_size) const;
    double PeriodicMinDistanceSq(const Vector& point, const Vector& box_size) const;

    /**
     * Computes maximum distance between boxes in periodic coordinate system
     */
    double PeriodicMaxDistanceSq(const DHrectBound& other, const Vector& box_size) const;
    double PeriodicMaxDistanceSq(const Vector& point, const Vector& box_size) const;

    double MaxDelta(const DHrectBound& other, double box_width, int dim) const;
    double MinDelta(const DHrectBound& other, double box_width, int dim) const;

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     */
    DRange RangeDistanceSq(const DHrectBound& other) const;

    /**
     * Calculates minimum and maximum bound-to-point squared distance.
     */
    DRange RangeDistanceSq(const Vector& point) const;

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
    double MinToMidSq(const DHrectBound& other) const;

    /**
     * Computes minimax distance, where the other node is trying to avoid me.
     */
    double MinimaxDistanceSq(const DHrectBound& other) const;

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    double MidDistanceSq(const DHrectBound& other) const;

    /**
     * Expands this region to include a new point.
     */
    DHrectBound& operator|=(const Vector& vector);

    /**
     * Expands this region to encompass another bound.
     */
    DHrectBound& operator|=(const DHrectBound& other);

    /**
     * Expand this bounding box to encompass another point. Done to 
     * minimize added volume in periodic coordinates.
     */
    DHrectBound& Add(const Vector&  other, const Vector& size);

    /**
     * Expand this bounding box in periodic coordinates, minimizing added volume.
     */
    DHrectBound& Add(const DHrectBound& other, const Vector& size);
};

#include "dhrectbound_impl.h"

#endif
